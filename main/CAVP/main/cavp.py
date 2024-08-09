import torch 
import torch.nn as nn
import importlib
from omegaconf import OmegaConf
from main.resnet.r2plus1d_18 import r2plus1d18KeepTemp
import torchvision.transforms as transforms

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class CAVPFeatureExtractor(torch.nn.Module):

    def __init__(self, config_path=None, ckpt_path=None):
        super(CAVPFeatureExtractor, self).__init__()

        # Initalize Stage1 CAVP model:
        print("Initalize Stage1 CAVP Model")
        config = OmegaConf.load(config_path)
        #self.stage1_model = instantiate_from_config(config.model).to(device)
        self.stage1_model = instantiate_from_config(config.model)

        # Loading Model from:
        assert ckpt_path is not None
        print("Loading Stage1 CAVP Model from: {}".format(ckpt_path))
        self.init_first_from_ckpt(ckpt_path)
        self.stage1_model.eval()
    
    def init_first_from_ckpt(self, path):
        # device = next(self.parameters()).device
        model = torch.load(path, map_location="cpu")
        # model = torch.load(path, map_location=device)
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        # Remove: module prefix
        new_model = {}
        for key in model.keys():
            new_key = key.replace("module.","")
            new_model[new_key] = model[key]
        missing, unexpected = self.stage1_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
    
    
    @torch.no_grad()
    def forward(self, x):
        if isinstance(x, tuple):
            frames, spec = x
            # print(f'frames shape: {frames.shape}')
            # print(f'spec shape: {spec.shape}')
            audio_contrastive_feats = self.stage1_model.encode_spec(spec, normalize=True, pool=False)
        else:
            frames = x
            audio_contrastive_feats = None

        frames = frames.permute(0, 2, 1, 3, 4) # Video: B x 3 x T x H x W -> B x T x 3 x H x W   
        video_contrastive_feats = self.stage1_model.encode_video(frames, normalize=True, pool=False)
        return video_contrastive_feats, audio_contrastive_feats
    
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    #     print("missing keys:")
    #     print(m)
    # if len(u) > 0 and verbose:
    #     print("unexpected keys:")
    #     print(u)
    model.cuda()
    model.eval()
    return model

class ResNet2plus1d(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet2plus1d, self).__init__()
        self.model = r2plus1d18KeepTemp(pretrained=pretrained)
        self.frames_transforms = transforms.Compose([
            transforms.Resize((112, 112), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        # x is (B, 3, T, 224,224), needs to be reshaped to (B, 3, T, 112, 112)
        B, C, T, H, W = x.shape
        x_transformed = []
        for b in range(B):
            for t in range(T):
                frame = x[b, :, t, :, :]
                frame = self.frames_transforms(frame)
                x_transformed.append(frame)
        x = torch.stack(x_transformed).view(B, C, T, 112, 112)
        print(f'shape of video after resize:', x.shape)
        # x = F.interpolate(x, size=(T, 112, 112), mode='trilinear', align_corners=False)
        x = self.model(x) #(B, 512, T)
        return x.permute(0, 2, 1) #(B, T, 512)