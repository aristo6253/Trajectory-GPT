import argparse
import os
import re
import imageio.v2 as imageio
from pathlib import Path

def get_sorted_rgb_images(folder):
    pattern = re.compile(r"step(\d+)$")
    subdirs = sorted(
        [d for d in os.listdir(folder) if pattern.match(d)],
        key=lambda d: int(pattern.search(d).group(1))
    )
    images = [os.path.join(folder, d, "rgb.png") for d in subdirs]
    images = [img for img in images if os.path.isfile(img)]
    return images

def make_mp4(images, output_path, fps):
    writer = imageio.get_writer(output_path, fps=fps)
    for img in images:
        writer.append_data(imageio.imread(img))
    writer.close()

def make_gif(images, output_path, fps):
    duration = 1.0 / fps
    frames = [imageio.imread(img) for img in images]
    imageio.mimsave(output_path, frames, format='GIF', duration=duration, loop=0)


def main():
    parser = argparse.ArgumentParser(description="Make MP4 and/or GIF from stepXX/rgb.png images.")
    parser.add_argument("folder", help="Folder containing stepXX subfolders.")
    parser.add_argument("--mp4", action="store_true", help="Generate MP4 video.")
    parser.add_argument("--gif", action="store_true", help="Generate GIF.")
    parser.add_argument("--fps", type=float, default=10, help="Frames per second (default: 10).")
    parser.add_argument("--output", type=str, default=None, help="Base name for output files.")

    args = parser.parse_args()

    images = get_sorted_rgb_images(args.folder)
    if not images:
        print("No valid rgb.png images found.")
        return

    folder_path = Path(args.folder)
    output_name = args.output or folder_path.stem
    output_base = folder_path / output_name

    if args.mp4:
        make_mp4(images, f"{output_base}.mp4", args.fps)
    if args.gif:
        make_gif(images, f"{output_base}.gif", args.fps)

if __name__ == "__main__":
    main()
