import os
from moviepy.editor import VideoFileClip

def convert_avi_to_mp4(source_folder, target_folder):

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.avi'):

                file_path = os.path.join(root, file)

                target_path = os.path.join(target_folder, os.path.basename(file).replace('.avi', '.mp4'))

                video_clip = VideoFileClip(file_path)
                if not os.path.exists(target_path):
                    video_clip.write_videofile(target_path, codec='libx264')

                video_clip.close()

source_folder = '/data/shaoshitong/UCF101/UCF-101/'
target_folder = '/data/shaoshitong/UCF101/UCF_real/'

convert_avi_to_mp4(source_folder, target_folder)