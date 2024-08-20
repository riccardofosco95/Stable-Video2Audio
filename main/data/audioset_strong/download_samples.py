import pandas as pd
import subprocess
import logging

def download_media_clips(csv_path, data_path):
    """Scarica clip video da YouTube utilizzando le informazioni da un CSV.

    Args:
        csv_path (str): Percorso del file CSV contenente gli ID dei video e i tempi di inizio.
        data_path (str): Percorso della directory dove salvare i clip scaricati.
    """

    logging.basicConfig(filename='download_log.txt', level=logging.INFO)

    try:
        df = pd.read_csv(csv_path, sep='\t')
    except FileNotFoundError:
        logging.error("File CSV non trovato: %s", csv_path)
        return

    processed_ids = set()

    for index, row in df.iterrows():
        segment_id = row['segment_id']
        if segment_id in processed_ids:
            logging.info(f"Segmento {segment_id} già processato, saltando.")
            continue

        processed_ids.add(segment_id)

        video_id, start_time = segment_id.rsplit('_', 1)
        url = f"https://www.youtube.com/watch?v={video_id}"
        print(url)

        yt_dlp_command = f"yt-dlp -f 'bestvideo+bestaudio/best' -g {url}"
        yt_dlp_process = subprocess.run(yt_dlp_command, shell=True, check=True, capture_output=True, text=True)
        video_url = yt_dlp_process.stdout.strip()

        # command = f"ffmpeg -ss {start_time} -t 10 -i $(yt-dlp -f 'bestvideo+bestaudio/best' -g {url}) -f mp4 -c:v copy -c:a copy {data_path}/{segment_id}.mp4"
        command = f"ffmpeg -ss {start_time} -t 10 -i {video_url} -f mp4 -c:v copy -c:a copy {data_path}/{segment_id}.mp4"

        try:
            logging.info(f"Scaricando segmento {index+1}: {segment_id}")
            subprocess.run(command, shell=True, check=True)
            logging.info(f"Segmento {segment_id} scaricato correttamente.")
        except subprocess.CalledProcessError as e:
            logging.warning(f"Errore durante il download del segmento {segment_id}: {e}")

if __name__ == "__main__":
    csv_path = "/homes/rfg543/Documents/Stable-Video2Audio/main/data/audioset_strong/audioset_eval_strong.tsv"
    data_path = "/homes/rfg543/Documents/Stable-Video2Audio/main/data/audioset_strong/output"
    download_media_clips(csv_path, data_path)
