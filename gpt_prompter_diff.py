import base64
import sys
import os
import re
import argparse
import cv2
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
    # parser.add_argument("--traj_file", type=str, help="File where the trajectory step will be saved")
    parser.add_argument("--incr_file", type=str, help="File where the trajectory increments will be saved")
    parser.add_argument("--logic_file", type=str, help="File where the trajectory logic will be saved")
    parser.add_argument("--overlay_cross", action="store_true", help="Overlay red cross on rgb.png")
    parser.add_argument('--preplanned_traj', type=str, default='NULL', help="Skip gpt for testing purposes")
    parser.add_argument('--test', type=str, default='full', choices=['full', 'cot', 'dnb', 'basic'], help="")
    args = parser.parse_args()

    traj_desc = args.traj_desc
    exp_name = args.exp_name

    if args.preplanned_traj == 'NULL':

        if args.test == 'full':
            print("FULL PROMPT")
            SYSTEM_PROMPT = gpt_params.SYSTEM_PROMPT_FULL
        elif args.test == 'cot':
            print("COT PROMPT")
            SYSTEM_PROMPT = gpt_params.SYSTEM_PROMPT_COT
        elif args.test == 'dnb':
            print("DNB PROMPT")
            SYSTEM_PROMPT = gpt_params.SYSTEM_PROMPT_DNB
        elif args.test == 'basic':
            print("BASIC PROMPT")
            SYSTEM_PROMPT = gpt_params.SYSTEM_PROMPT_BASIC
        
        client = OpenAI(api_key=api_key.OPENAI_API)

        base_dir = os.path.join("results_diff", exp_name)
        max_step, step_path = get_latest_step_folder(base_dir)

        rgb0_path = os.path.join(base_dir, 'step00', "rgb.png")
        rgb_path = os.path.join(step_path, "rgb.png")
        if args.test in ['full', 'dnb']:
            depth_path = os.path.join(step_path, "depth.png")
            bev_path = os.path.join(step_path, "bev.png")
        prompt_response_path = os.path.join(step_path, "prompt_and_response.txt")
        response_hist_path = os.path.join(base_dir, "response_history.txt")

        if args.overlay_cross:
            guided_rgb_path = os.path.join(step_path, "rgb_guided.png")
            overlay_red_cross(rgb_path, guided_rgb_path)


        traj_hist = ""

        if max_step > 0:
            with open(args.incr_file, "r") as f_incr, open(args.logic_file, "r") as f_logic:
                incr_lines = f_incr.readlines()
                logic_lines = f_logic.readlines()

                start_idx = max(0, len(incr_lines) - 5)
                recent = zip(incr_lines[start_idx:], logic_lines[start_idx:])

                traj_hist = "\n".join(
                    [f"Step{start_idx + i + 1}: {incr.strip()} ({logic.strip() if args.test in ['full', 'cot'] else ''})"
                     for i, (incr, logic) in enumerate(recent)]
                )
        else:
            traj_hist = "(No Steps yet)"

        # Encode images
        rgb0_b64 = encode_image(rgb0_path)
        rgb_b64 = encode_image(guided_rgb_path) if args.overlay_cross else encode_image(rgb_path)
        if args.test in ['full', 'dnb']:
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
4. Objective in format: ###\n(Objective)\n### (no indents)
5. Motion command in format: `dx dy dz dyaw dpitch droll`
Follow camera-centric conventions exactly. No extra text.
"""
        print(f"{full_user_prompt = }")

        user_content = [
            {"type": "text", "text": full_user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb0_b64}"}},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}},
        ]

        if args.test in ['full', 'dnb']:
            user_content.extend([
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{depth_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{bev_b64}"}},
            ])

        # Create the prompt and image inputs
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_content}
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
        
        if args.test in ['full', 'cot']:
            if logic:
                with open(args.logic_file, "a") as f:
                    f.write(logic.group(1).strip() + "\n")
            else:
                raise ValueError("No logic command found in GPT response.")

        with open(prompt_response_path, "w") as f:
            f.write("=== Prompt ===\n")
            f.write(full_user_prompt.strip() + "\n\n")
            f.write("=== Response ===\n")
            f.write(response.choices[0].message.content.strip() + "\n")

        with open(response_hist_path, "a") as f:
            f.write(f"\n=== Response {max_step} ===\n")
            f.write(text.strip() + "\n")
    else:
        base_dir = os.path.join("results_diff", exp_name)
        max_step, _ = get_latest_step_folder(base_dir)
        step_idx = max_step# + 1

        # rgb_path = os.path.join(step_path, "rgb.png")
        # if args.overlay_cross:
        #     guided_rgb_path = os.path.join(step_path, "rgb_guided.png")
        #     overlay_red_cross(rgb_path, guided_rgb_path)

        with open(args.preplanned_traj, "r") as f:
            lines = f.readlines()

        if step_idx >= len(lines):
            raise IndexError(f"Preplanned trajectory does not contain step {step_idx}")

        incr_line = lines[step_idx].strip()
        if not incr_line:
            raise ValueError(f"Preplanned trajectory at step {step_idx} is empty")

        with open(args.incr_file, "a") as f:
            f.write(incr_line + "\n")

        with open(args.logic_file, "a") as f:
            f.write(f"(Preplanned step {step_idx})\n")


if __name__ == "__main__":
    main()
