import base64
import sys
import os
import re
import argparse
import cv2
from openai import OpenAI
import api_key
import gpt_params
import numpy as np
import json
from utils import *

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

def overlay_red_cross(image_path, output_path, thickness=2):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at {image_path}")

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # Draw red lines
    cv2.line(img, (cx, 0), (cx, h - 1), (0, 0, 255), thickness=thickness)
    cv2.line(img, (0, cy), (w - 1, cy), (0, 0, 255), thickness=thickness)

    # Overwrite original image
    cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(description="6-DoF trajectory planning prompt")
    parser.add_argument("--traj_desc", type=str, help="Trajectory description")
    parser.add_argument("--exp_name", type=str, help="Experiment name under results/")
    parser.add_argument("--traj_json", type=str, help="File where the trajectory step will be saved")
    parser.add_argument("--incr_file", type=str, help="File where the trajectory increments will be saved")
    parser.add_argument("--logic_file", type=str, help="File where the trajectory increments will be saved")
    parser.add_argument("--overlay_cross", action="store_true", help="Overlay red cross on rgb.png")
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    traj_desc = args.traj_desc
    exp_name = args.exp_name

    client = OpenAI(api_key=api_key.OPENAI_API)

    base_dir = os.path.join("results_3dgs", exp_name)
    max_step, step_path = get_latest_step_folder(base_dir)

    rgb0_path = os.path.join(base_dir, 'step00', "rgb.png")
    rgb_path = os.path.join(step_path, "rgb.png")
    depth_path = os.path.join(step_path, "depth.png")
    bev_path = os.path.join(step_path, "bev.png")
    prompt_response_path = os.path.join(step_path, "prompt_and_response.txt")
    response_hist_path = os.path.join(base_dir, "response_history.txt")

    # print(f"{gpt_params.SYSTEM_PROMPT.strip() =  }")

    if args.overlay_cross:
        guided_rgb_path = os.path.join(step_path, "rgb_guided.png")
        overlay_red_cross(rgb_path, guided_rgb_path)


    traj_hist = ""

    if max_step > 0:
        with open(args.incr_file, "r") as f_incr, open(args.logic_file, "r") as f_logic:
            incr_lines = f_incr.readlines()
            logic_lines = f_logic.readlines()
            traj_hist = "\n".join(
                [f"Step{i+1}: {incr.strip()} ({logic.strip()})"
                for i, (incr, logic) in enumerate(zip(incr_lines, logic_lines))]
            )
    else:
        traj_hist = "(No Steps yet)"

    # Encode images
    rgb0_b64 = encode_image(rgb0_path)
    rgb_b64 = encode_image(guided_rgb_path) if args.overlay_cross else encode_image(rgb_path)
    depth_b64 = encode_image(depth_path)
    bev_b64 = encode_image(bev_path)


    full_user_prompt = f"""Trajectory Step {max_step} — Plan the next move.

Goal:
{traj_desc}

Step History:
{traj_hist}

Reminder: Respond with:
1. Have some continuity reasoning
2. Trajectory reasoning (max 4 lines)
3. Checklist Verification
4. Objective in format: ###\n(Objective)\n###
5. Motion command in format: `dx dy dz dyaw dpitch droll`
Follow camera-centric conventions exactly. No extra text.
"""
    print(f"{full_user_prompt.strip() = }")

    # Create the prompt and image inputs
    messages = [
        {"role": "system", "content": gpt_params.SYSTEM_PROMPT.strip()},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": full_user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb0_b64}"}},
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
    step = re.search(r"```(?:[^\n]*)\n(.*?)\n```", text, re.DOTALL)
    logic = re.search(r"###(?:[^\n]*)\n(.*?)\n###", text, re.DOTALL)
    # print(f"{match = }")
    if step:
        with open(args.incr_file, "a") as f:
            f.write(step.group(1).strip() + "\n")
    else:
        raise ValueError("No motion command found in GPT response.")
    
    if logic:
        with open(args.logic_file, "a") as f:
            f.write(logic.group(1).strip() + "\n")
    else:
        raise ValueError("No logic command found in GPT response.")


    ### Here we have the next step
    ### We need to fetch the last extrinsic present in the json
    last_pose_info = get_last_pose_info(args.traj_json)
    R_prev = last_pose_info["R"]
    t_prev = last_pose_info["t"]
    ### Combine increment and extrinsic in order to find the next extrinsic
    incr_vals = list(map(float, step.group(1).strip().split()))
    dx, dy, dz, dyaw, dpitch, droll = incr_vals
    T_delta = extrinsic_matrix(dyaw, dpitch, -droll, [dx, -dy, dz])

    T_prev = np.eye(4)
    T_prev[:3, :3] = R_prev
    T_prev[:3, 3] = t_prev.flatten()

    T_next = T_prev @ T_delta
    R_next = T_next[:3, :3]
    t_next = T_next[:3, 3].reshape(3, 1)

    ### Save the new extrinsic in the json
    with open(args.traj_json, "r") as f:
        traj_data = json.load(f)

    new_pose = {
        "id": last_pose_info["id"] + 1,
        "img_name": f"step{max_step+1:02}/rgb.png",
        "width": last_pose_info["width"],
        "height": last_pose_info["height"],
        "position": t_next.flatten().tolist(),
        "rotation": R_next.tolist(),
        "fx": last_pose_info["fx"],
        "fy": last_pose_info["fy"]
    }

    traj_data.append(new_pose)

    with open(args.traj_json, "w") as f:
        json.dump(traj_data, f, indent=4)

    with open(f"{args.model}/{args.exp_name}.json", "w") as f:
        json.dump(traj_data, f, indent=4)



    with open(prompt_response_path, "w") as f:
        f.write("=== Prompt ===\n")
        f.write(full_user_prompt.strip() + "\n\n")
        f.write("=== Response ===\n")
        f.write(response.choices[0].message.content.strip() + "\n")

    with open(response_hist_path, "a") as f:
        f.write(f"\n=== Response {max_step} ===\n")
        f.write(text.strip() + "\n")

if __name__ == "__main__":
    main()
