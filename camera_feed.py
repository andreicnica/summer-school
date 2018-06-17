import torch
import cv2
import numpy as np
from argparse import Namespace
from termcolor import colored as clr
import datetime
import os
import argparse

WIDTH = 24
HEIGHT = 28
ZONE_FACTOR = 2


# Useful functions
def print_info(message):
    print(clr("[MAIN] ", "yellow") + message)


def print_colored(info, variable, color):
    print("{} {}".format(clr(info, color), variable))

    """
    Set optional checkpoint load path 
    """

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Live test.')
    parser.add_argument("-chk", '--checkpoint_path', type=str,
                        dest="checkpoint", default=False,
                        help='path to checkpoint save file')

    args = parser.parse_args()
    if "checkpoint" in args:
        checkpoint_load = args.checkpoint
    else:
        checkpoint_load = False

    # ============================================

    config = read_config()
    print_info("Startig train script: {}".format(datetime.datetime.now()))

    # Create Data (local storage of important data)
    data = Namespace()

    experiment_cfg = config.experiment
    use_cuda = config.general.use_cuda
    batch_size = config.general.batch_size

    model = get_model(config.model.name)(config.model)
    if use_cuda:
        model.cuda()

    data_cfg = config.dataset
    mean = torch.FloatTensor(data_cfg.mean).unsqueeze(1).unsqueeze(2) \
        .expand((3, HEIGHT, WIDTH))
    std = torch.FloatTensor(data_cfg.std).unsqueeze(1).unsqueeze(2) \
        .expand((3, HEIGHT, WIDTH))
    if use_cuda:
        mean = mean.cuda()
        std = std.cuda()

    # ============================================
    # model_path = config.experiment.resume
    if checkpoint_load:
        model_path = checkpoint_load
    else:
        model_path = config.experiment.resume

    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)

        print_colored("=> loading checkpoint: ", model_path, "red")
        print_colored("===> Evaluation prec1 avg: ",
                      checkpoint["eval_prec1"].avg, "red")

        # ==================================================================
        # -- Partial model arch
        model_dict = model.state_dict()
        pretrained_dict = checkpoint['model']
        diff = {k: v for k, v in model_dict.items() if \
                k in pretrained_dict and pretrained_dict[
                    k].size() != v.size()}
        pretrained_dict.update(diff)

        state = model.state_dict()
        state.update(pretrained_dict)
        model.load_state_dict(state)
        # ==================================================================

    model.eval()

    cam = cv2.VideoCapture(0)
    ret, frame = cam.read()
    cv2.imshow("Frame", frame); cv2.waitKey(0); cv2.destroyAllWindows()

    zone_factor = ZONE_FACTOR
    zone_offset = np.array([WIDTH, HEIGHT]) * zone_factor

    frame_no = 0
    test_loss = torch.autograd.Variable(torch.LongTensor([1]))
    if use_cuda:
        test_loss = test_loss.cuda()
    while ret:
        zone_offset = (np.array([WIDTH, HEIGHT]) * zone_factor).astype(int)

        ret, frame = cam.read()

        frame_show = frame.copy()
        frame_show = cv2.flip(frame_show, 1)

        center = (np.array([frame.shape[1], frame.shape[0]]) / 2).astype(int)
        p1 = center - zone_offset
        p2 = center + zone_offset

        scan = frame[p1[1]: p2[1], p1[0]:p2[0], :]
        cv2.imshow("Scan", scan)
        scan = cv2.resize(scan, (WIDTH, HEIGHT))
        scan = cv2.cvtColor(scan, cv2.COLOR_BGR2RGB)

        scan = torch.FloatTensor(scan.astype(float))
        if use_cuda:
            scan = scan.cuda()

        # transform to RGB
        scan = scan.transpose(2, 0).transpose(1, 2)
        scan.div_(255)
        # print(scan)

        scan = (scan - mean) / std
        res = model(torch.autograd.Variable(scan.unsqueeze(0)))
        if res.data[0][0] > res.data[0][1]:
            cv2.rectangle(frame_show, tuple(p1), tuple(p2), (0, 255, 0),
                          thickness=2)
        else:
            cv2.rectangle(frame_show, tuple(p1), tuple(p2), (0, 0, 255),
                          thickness=2)
        cv2.imshow("GRAB", frame_show)

        # print(torch.nn.CrossEntropyLoss()(res, test_loss))
        print(res)
        key = cv2.waitKey(1)
        if chr(key % 256) == ",":
            zone_factor += 0.1
        elif chr(key % 256) == ".":
            zone_factor -= 0.1

        frame_no += 1
