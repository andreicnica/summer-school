"""
    Code adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
"""

import torch
import cv2
import numpy as np
import os
import argparse
import time
import torchvision.transforms as transforms
import torch
import sys
import re
from models import get_model

FULL_CAPTURE_WIN = "Full_capture"
CROP_WIN = "Crop"
COLOR_CROP = (0, 0, 255)

RECORD_COLOR = [(0, 255, 0), (0, 0, 255)]
REC_POSITION = (50, 80)
REC_RADIUS = 20
LABEL_POSITION = (5, 25)
FONT = cv2.FONT_HERSHEY_TRIPLEX
FONT_SIZE = 0.6
FONT_CLR = 255

ACTION_KEYS = {
    "zoom_in": {"key": ",", "info": "Increase crop size."},
    "zoom_out": {"key": ".", "info": "Decrease crop size."},
    "label": {"key": "[0-9]", "info": "Set label keys."},
    "record": {"key": "x", "info": "Toggle record on/off."},
    "quit": {"key": "q", "info": "Exit script."},
}


def get_img_path(base_folder: str, label: int, img_name: str):
    fld = os.path.join(base_folder, str(label))
    if not os.path.isdir(fld):
        os.mkdir(fld)
    return os.path.join(fld, img_name)


def clear_line(no_lines: int):
    for i in range(no_lines):
        sys.stdout.write("\033[F")  # back to previous line
        sys.stdout.write("\033[K")  # clear line


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Live test & data gathering.')
    parser.add_argument("-chk", '--checkpoint_path', type=str,
                        dest="checkpoint", default=False,
                        help='Path to model checkpoint.')
    parser.add_argument('--save-folder', type=str,
                        dest="save_folder", default="datasets/",
                        help='Save folder of collected images.')
    parser.add_argument('--zone-factor', type=float, default=0.5,
                        help='Percentage of full view image to crop.')
    parser.add_argument('--zoom-factor-step', type=float, default=0.05,
                        help='Zoom factor.')
    parser.add_argument('--full-view-size', type=int, default=512,
                        help='Size in px of "full view" window.')
    parser.add_argument('--crop-view-size', type=int, default=256,
                        help='Size in px of "crop" window.')
    parser.add_argument('--camera-id', type=int, default=0, help='Initial crop view port size.')

    # ==============================================================================================
    # -- Load arguments

    args = parser.parse_args()
    checkpoint_load = False
    if "checkpoint" in args:
        checkpoint_load = args.checkpoint
    save_folder = args.save_folder

    camera_id = args.camera_id
    zone_factor = args.zone_factor
    view_size = args.full_view_size
    crop_view_size = args.crop_view_size
    zoom_factor_step = args.zoom_factor_step

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==============================================================================================
    # -- Load model & classification data
    model = None
    transform = None

    if checkpoint_load:
        checkpoint = torch.load(checkpoint_load)
        transform = checkpoint["data_transforms"]["val"]
        transform = transforms.Compose([transforms.ToPILImage()] + transform.transforms)
        model_name = checkpoint["model_name"]
        in_size = checkpoint["in_size"]
        out_size = checkpoint["out_size"]
        state_dict = checkpoint["state_dict"]
        model = get_model(model_name, in_size=in_size, out_size=out_size,
                          pretrained=True, model_weights=state_dict)

        model = model.to(device)
        model.eval()

    # ==============================================================================================
    # -- Data saving

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    print(f"Saving data to: {save_folder}")

    imgs_collected = {}
    img_format = f"{int(time.time())}_{{}}.png"

    # ==============================================================================================
    # -- Start camera feed and initialize capturing information

    cam = cv2.VideoCapture(camera_id)
    ret, frame = cam.read()
    assert ret, "Cannot read from camera."

    view_scale = view_size / float(frame.shape[1])

    cv2.namedWindow(FULL_CAPTURE_WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CROP_WIN, cv2.WINDOW_NORMAL)

    # ==============================================================================================
    # -- Print information
    keys_msg = "\t ||  ".join([f'[{x["key"]}] {x["info"]}' for k, x in ACTION_KEYS.items()])
    print_info = f"{keys_msg}\nCollected data: {{}}\nSelected label: {{}}\n" \
                 f"Recording: {{}}\nPredictions: {{}}"

    # ==============================================================================================
    # -- Useful variables
    frame_no = 0
    crt_label = 0
    last_msg_len = 0
    predicted = -1

    recording = False
    predictions = None

    keys = argparse.Namespace()
    keys.__dict__.update({k: v["key"] for k, v in ACTION_KEYS.items()})

    print("=" * 80)
    while ret:
        zone_offset = (np.array([frame.shape[0], frame.shape[0]]) * zone_factor / 2.).astype(int)

        ret, frame = cam.read()

        # Get crop from original frame
        center = (np.array([frame.shape[1], frame.shape[0]]) / 2.).astype(int)

        p1 = center - zone_offset
        p2 = center + zone_offset

        scan = frame[p1[1]: p2[1], p1[0]:p2[0], :]

        # Record data
        if recording:
            if crt_label in imgs_collected:
                imgs_collected[crt_label] += 1
            else:
                imgs_collected[crt_label] = 0
            img_path = get_img_path(save_folder, crt_label,
                                    img_format.format(imgs_collected[crt_label]))
            cv2.imwrite(img_path, scan)

        # Classify crop
        if checkpoint_load:
            # Transform to RGB
            with torch.no_grad():
                in_data = transform(scan.transpose((2, 0, 1))).unsqueeze(0)
                in_data = in_data.to(device)

                output = model(in_data)
                output = output.cpu()
                _, predicted = torch.max(output, 1)

            predicted = predicted[0].item()
            predictions = output.numpy()

        # Prepare full_view image (resize and draw rectangle)
        frame_show = frame.copy()
        frame_show = cv2.flip(frame_show, 1)
        frame_show = cv2.resize(frame_show, (0, 0), fx=view_scale, fy=view_scale)
        p1_scaled = (p1 * view_scale).astype(np.uint)
        p2_scaled = (p2 * view_scale).astype(np.uint)
        frame_show = cv2.rectangle(frame_show, tuple(p1_scaled), tuple(p2_scaled), COLOR_CROP)

        clr = RECORD_COLOR[int(recording)]
        frame_show = cv2.putText(frame_show, f"Selected label: {crt_label}",
                                 LABEL_POSITION, FONT, FONT_SIZE, FONT_CLR)
        frame_show = cv2.circle(frame_show, REC_POSITION, REC_RADIUS, clr, thickness=-1)

        # Prepare crop
        crop_show = scan.copy()
        crop_show = cv2.flip(crop_show, 1)
        crop_show = cv2.resize(crop_show, (crop_view_size, crop_view_size))
        if checkpoint_load:
            crop_show = cv2.putText(crop_show, f"{predicted}",
                                    LABEL_POSITION, FONT, FONT_SIZE, FONT_CLR)

        cv2.imshow(FULL_CAPTURE_WIN, frame_show)
        cv2.imshow(CROP_WIN, crop_show)

        key = cv2.waitKey(1)
        if chr(key % 256) == keys.zoom_out:
            zone_factor += zoom_factor_step
        elif chr(key % 256) == keys.zoom_in:
            zone_factor -= zoom_factor_step
        elif ord("0") <= key % 256 <= ord("9"):
            crt_label = int(chr(key % 256))
        elif chr(key % 256) == keys.record:
            recording = not recording
        elif chr(key % 256) == keys.quit:
            break
        else:
            pass
            # print(f"Unrecognized key: {key % 256}")

        frame_no += 1

        clear_line(last_msg_len)
        sys.stdout.flush()

        collected_data_msg = "\t".join([f"[{k}] {v}" for k, v in imgs_collected.items()])
        msg = print_info.format(collected_data_msg, crt_label, recording, predictions)
        print(msg)
        last_msg_len = len(re.findall("\n", msg)) + 1
