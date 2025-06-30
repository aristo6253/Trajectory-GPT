from moviepy.editor import VideoFileClip
import sys
import os

def mp4_to_gif(input_path, output_path=None, start=0, end=None, height=320, fps=10):
    clip = VideoFileClip(input_path)
    if end:
        clip = clip.subclip(start, end)
    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + ".gif"
    clip.write_gif(output_path, fps=fps)
    print(f"Saved GIF to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mp4_to_gif.py <input_video> [output_gif]")
        sys.exit(1)
    input_video = sys.argv[1]
    output_gif = sys.argv[2] if len(sys.argv) > 2 else None
    mp4_to_gif(input_video, output_gif)
