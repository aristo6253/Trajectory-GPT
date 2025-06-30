import base64
import sys
import os
import re
import argparse
from openai import OpenAI
import api_key
import gpt_params

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def get_latest_step_folder(base_dir):
    step_folders = [
        f for f in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, f)) and re.match(r"step\d\d", f)
    ]
    if not step_folders:
        raise ValueError(f"No step folders found in {base_dir}")
    step_folders.sort()
    max_step_folder = step_folders[-1]
    max_step = int(max_step_folder[-2:])  # from "stepXX"
    return max_step, os.path.join(base_dir, max_step_folder)

def main():
    parser = argparse.ArgumentParser(description="6-DoF trajectory planning prompt")
    parser.add_argument("--traj_desc", type=str, help="Trajectory description")
    parser.add_argument("--exp_name", type=str, help="Experiment name under results/")
    parser.add_argument("--traj_file", type=str, help="File where the trajectory step will be saved")
    args = parser.parse_args()

    traj_desc = args.traj_desc
    exp_name = args.exp_name

    client = OpenAI(api_key=api_key.OPENAI_API)

    base_dir = os.path.join("results", exp_name)
    max_step, step_path = get_latest_step_folder(base_dir)

    rgb_path = os.path.join(step_path, "rgb.png")
    depth_path = os.path.join(step_path, "depth.png")
    bev_path = os.path.join(step_path, "bev.png")

    # Encode images
    rgb_b64 = encode_image(rgb_path)
    depth_b64 = encode_image(depth_path)
    bev_b64 = encode_image(bev_path)


    full_user_prompt = f"""Trajectory Step {max_step} — Plan the next move.

Goal:
{traj_desc}

Reminder: Respond with:
1. Step-by-step reasoning (max 4 lines)
2. Motion command in format: `dx dy dz dyaw dpitch droll`
Follow camera-centric conventions exactly. No extra text.
"""
    print(f"{full_user_prompt = }")

    # Create the prompt and image inputs
    messages = [
        {"role": "system", "content": gpt_params.SYSTEM_PROMPT2.strip()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{depth_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bev_b64}"}},
            ]
        }
    ]


    # Send request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    # Output response
    print(response.choices[0].message.content)

    text = response.choices[0].message.content
    match = re.search(r"```(?:[^\n]*)\n(.*?)\n```", text, re.DOTALL)
    print(f"{match = }")
    if match:
        with open(args.traj_file, "a") as f:
            f.write(match.group(1).strip() + "\n")

if __name__ == "__main__":
    main()
