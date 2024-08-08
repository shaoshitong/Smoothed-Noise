from moviepy.editor import *
import glob
"""
mkdir 1
mkdir 2
mkdir 3
cp -r *.mp4 1/
cp -r *.mp4 2/
cp -r *.mp4 3/
"""
def convert_mp4_to_gif(mp4_path, gif_path, duration=None, fps=10):
    """
    Convert an MP4 video to a GIF.

    :param mp4_path: Path to the MP4 file.
    :param gif_path: Path where the GIF will be saved.
    :param duration: Duration (in seconds) of the GIF. If None, the whole video is used.
    :param fps: Frames per second for the GIF.
    """
    # Load the video
    video = VideoFileClip(mp4_path)

    # If a duration is specified, subclip the video
    if duration is not None:
        video = video.subclip(0, duration)

    # Write the video to a GIF
    video.write_gif(gif_path, fps=fps)

    print(f"GIF saved to {gif_path}")

# Example usage
mp4_file = 'example.mp4'
gif_file = 'output.gif'

mp4_paths = glob.glob(r"./*.mp4")
for mp4_path in mp4_paths:
    gif_path = mp4_path.replace(".mp4",".gif")
    convert_mp4_to_gif(mp4_path, gif_path, duration=20, fps=20)