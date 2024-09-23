import os
import glob
import torch
from PIL import Image
from natsort import natsorted
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
import cv2
import numpy as np
import shutil   


weights = Raft_Large_Weights.DEFAULT
transforms_raft = weights.transforms()

def read_image_and_apply_transforms(frame_list):
    imgs = []
    for img_path in frame_list:
        image = Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image)
        imgs.append(image.unsqueeze(0))
    imgs = torch.cat(imgs, dim=0).squeeze()
    return imgs

device = "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
model = model.eval()

# root_dir = "/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed"

# frames_transforms = transforms.Compose([
#     transforms.Resize((320, 240), antialias=True),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

def preprocess(frame1, frame2):
    # frame1 = transforms.Resize((320, 240), antialias=True)(frame1)
    # frame2 = transforms.Resize((320, 240), antialias=True)(frame2)
    return transforms_raft(frame1, frame2)

root_dir = "/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed-25fps-441kHz"
# video_dir = "2015-02-16-16-49-06"
# video_dir_path = os.path.join(root_dir, video_dir, "frames")
# flow_dir_path = os.path.join(root_dir, video_dir, "optical_flow")

# os.makedirs(flow_dir_path, exist_ok=True)   
# frame_files = natsorted(glob.glob(f"{video_dir_path}/*.jpg"))
# frames = read_image_and_apply_transforms(frame_files)

# for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
#     # Note: it would be faster to predict batches of flows instead of individual flows
#     img1, img2 = preprocess(img1, img2)

#     img1 = img1.unsqueeze(0)
#     img2 = img2.unsqueeze(0)
#     list_of_flows = model(img1.to(device), img2.to(device))
#     predicted_flow = list_of_flows[-1][0]

#     # Convert the flow to numpy arrays
#     flow_x = predicted_flow[0].detach().cpu().numpy()
#     flow_y = predicted_flow[1].detach().cpu().numpy()

#     # Normalize the flow to 0-255 for visualization as grayscale
#     flow_x = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
#     flow_y = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

#     # Convert the flow to uint8
#     flow_x = np.uint8(flow_x)
#     flow_y = np.uint8(flow_y)

#     # Save the flow components as grayscale images
#     filename_x = "{}.predicted_flow_x_{:06}.jpg".format(video_dir, i+1)
#     filename_y = "{}.predicted_flow_y_{:06}.jpg".format(video_dir, i+1)
#     cv2.imwrite(os.path.join(flow_dir_path, filename_x), flow_x)
#     cv2.imwrite(os.path.join(flow_dir_path, filename_y), flow_y)


for video_dir in os.listdir(root_dir):
    print(f"Processing video: {video_dir}")
    video_dir_path = os.path.join(root_dir, video_dir, "frames")
    flow_dir_path = os.path.join(root_dir, video_dir, "optical_flow")

    if os.path.isdir(video_dir_path):
            if os.path.exists(flow_dir_path):
                print(f"Optical flow for video {video_dir} already exists. Skipping.")
                continue

            os.makedirs(flow_dir_path, exist_ok=True)   
            frame_files = natsorted(glob.glob(f"{video_dir_path}/*.jpg"))
            # frame_files = frame_files[:250]

            # for file in frame_files:
            #     shutil.copy(file, flow_dir_path)

            frames = read_image_and_apply_transforms(frame_files)
            # print(frames.shape)

            for i, (img1, img2) in enumerate(zip(frames, frames[1:])):
                # Note: it would be faster to predict batches of flows instead of individual flows
                img1, img2 = preprocess(img1, img2)

                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
                list_of_flows = model(img1.to(device), img2.to(device))
                predicted_flow = list_of_flows[-1][0]

                # Convert the flow to numpy arrays
                flow_x = predicted_flow[0].detach().cpu().numpy()
                flow_y = predicted_flow[1].detach().cpu().numpy()

                # Normalize the flow to 0-255 for visualization as grayscale
                flow_x = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
                flow_y = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)

                # Convert the flow to uint8
                flow_x = np.uint8(flow_x)
                flow_y = np.uint8(flow_y)

                # Save the flow components as grayscale images
                filename_x = "{}.predicted_flow_x_{:06}.jpg".format(video_dir, i+1)
                filename_y = "{}.predicted_flow_y_{:06}.jpg".format(video_dir, i+1)
                cv2.imwrite(os.path.join(flow_dir_path, filename_x), flow_x)
                cv2.imwrite(os.path.join(flow_dir_path, filename_y), flow_y)

                # If this is the last iteration, save the optical flow again
                if i == len(frames) - 2:  # -2 because we start from 0 and we have frames[1:]
                    filename_x = "{}.predicted_flow_x_{:06}.jpg".format(video_dir, i+2)
                    filename_y = "{}.predicted_flow_y_{:06}.jpg".format(video_dir, i+2)
                    cv2.imwrite(os.path.join(flow_dir_path, filename_x), flow_x)
                    cv2.imwrite(os.path.join(flow_dir_path, filename_y), flow_y)

# import os
# import shutil

# root_dir = "/import/c4dm-datasets-ext/DIFF-SFX/GREATEST-HITS-DATASET/mic-mp4-processed"

# for video_dir in os.listdir(root_dir):
#     flow_dir_path = os.path.join(root_dir, video_dir, "optical_flow")
#     if os.path.exists(flow_dir_path):
#         print(f"Deleting optical flow directory in {video_dir}")
#         shutil.rmtree(flow_dir_path)
#     else:
#         print(f"No optical flow directory found in {video_dir}")