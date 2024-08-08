import os
from moviepy.editor import VideoFileClip
import shutil

def choose_video(source_folder, target_folder, choose_number):

    assert os.path.exists(target_folder) and os.path.exists(source_folder), \
        f"Make sure that {source_folder} and {target_folder} are existed!"

    total_files = []
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            if file.endswith('.mp4'):
                total_files.append(file)
    
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.mp4') and (file not in total_files):
                file_path = os.path.join(root, file)
                os.remove(file_path)

if __name__ == "__main__":
    source_folder = '/data/shaoshitong/UCF101/UCF_fake_v2/'
    target_folder = '/data/shaoshitong/UCF101/UCF_real_2/'
    choose_video(source_folder, target_folder, 500)