import os
from moviepy.editor import VideoFileClip
import shutil

def choose_video(source_folder, target_folder, choose_number, source_folder_2, target_folder_2):

    assert os.path.exists(target_folder) and os.path.exists(source_folder), \
        f"Make sure that {source_folder} and {target_folder} are existed!"
    
    if not os.path.exists(source_folder_2):
        os.makedirs(source_folder_2, exist_ok=True)
    if not os.path.exists(target_folder_2):
        os.makedirs(target_folder_2, exist_ok=True)

    file_paths = []
    t_file_paths = []
    total_files = []
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
                t_file_path = os.path.join(target_folder, file)
                t_file_paths.append(t_file_path)
                total_files.append(file)
    
    file_paths = file_paths[:choose_number]
    t_file_paths = t_file_paths[:choose_number]
    total_files = total_files[:choose_number]
    
    for file_path, t_file_path, total_file in zip(file_paths, t_file_paths, total_files):
        file_path_2 = os.path.join(source_folder_2,total_file)
        t_file_path_2 = os.path.join(target_folder_2,total_file)
        shutil.move(file_path, file_path_2)
        shutil.move(t_file_path, t_file_path_2)
        print(f"Successfully move {file_path} to {file_path_2}")
    return file_paths, t_file_paths

if __name__ == "__main__":
    source_folder = '/data/shaoshitong/UCF101/UCF_fake/'
    target_folder = '/data/shaoshitong/UCF101/UCF_real/'

    source_folder_2 = '/data/shaoshitong/UCF101/UCF_fake_2/'
    target_folder_2 = '/data/shaoshitong/UCF101/UCF_real_2/'
    
    choose_video(source_folder, target_folder, 500, source_folder_2, target_folder_2)