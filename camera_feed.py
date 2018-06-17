import torch
import cv2
import numpy as np
import os
import argparse
import time
import torchvision.transforms as transforms
import torch

FULL_CAPTURE_WIN = "Full_capture"
CROP_WIN = "Crop"
COLOR_CROP = (0, 0, 255)

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
    parser.add_argument('--full-view-size', type=int, default=512,
                        help='Size in px of "full view" window.')
    parser.add_argument('--crop-view-size', type=int, default=512,
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
    view_size = args.view_size

    # ==============================================================================================
    # -- Load model
    model: torch.nn.Module = None
    transform: transforms.Compose = None

    if checkpoint_load:
        # TODO Load model and weights
        pass
    transforms.RG
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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

    frame_no = 0
    while ret:
        zone_offset = (np.array([frame.shape[0], frame.shape[1]]) * zone_factor).astype(int)

        ret, frame = cam.read()

        frame_show = frame.copy()
        frame_show = cv2.flip(frame_show, 1)

        # -- Get crop
        center = (np.array([frame.shape[1], frame.shape[0]]) / 2).astype(int)
        p1 = center - zone_offset
        p2 = center + zone_offset
        scan = frame[p1[1]: p2[1], p1[0]:p2[0], :]

        # Color crop in full view

        if checkpoint_load:
            # Transform to RGB
            with torch.no_grad:
                in_data = transform(scan.transpose((2, 0, 1))).unsqueeze(0)
                output = model(in_data)
                _, predicted = torch.max(output, 1)

        cv2.imshow(FULL_CAPTURE_WIN, frame_show)

        # print(torch.nn.CrossEntropyLoss()(res, test_loss))
        print(res)
        key = cv2.waitKey(1)
        if chr(key % 256) == ",":
            zone_factor += 0.1
        elif chr(key % 256) == ".":
            zone_factor -= 0.1

        frame_no += 1
