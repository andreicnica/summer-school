import os
import argparse
import glob
import time
import shutil
import numpy as np
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Shuffle data and split data.')
    parser.add_argument('--data-folder', type=str, default="datasets/")
    parser.add_argument('--split', type=float, nargs='+', default=[0.8, 0.2])
    parser.add_argument('--split-names', type=str, nargs='+', default=["train", "val"])
    parser.add_argument('--extensions', type=str, nargs='+', default=["png"])
    parser.add_argument('--overwrite-duplicates', default=False, action='store_true')

    args = parser.parse_args()

    base_path = args.data_folder
    extensions = args.extensions
    overwrite_duplicates = args.overwrite_duplicates
    split = args.split
    split_names = args.split_names

    assert len(split_names) == len(split), "Length of split factors must be equal to split names."
    assert os.path.isdir(base_path), "No folder with this name"

    # -- Get all files and group by label
    # Assumes label name last folder name
    file_paths = []
    for ext in extensions:
        file_paths.extend(glob.glob(f"{base_path}/**/*.{ext}", recursive=True))

    # -- Move to temporary directory and group by label
    sub_directories = glob.glob(f'{base_path}/*')
    tmp_dir = os.path.join(base_path, f"tmp_{int(time.time())}")
    assert not os.path.isdir(tmp_dir), "Temp folder already exists."
    os.mkdir(tmp_dir)

    dataset = {}
    for file in file_paths:
        label = os.path.basename(os.path.dirname(file))
        file_name = os.path.basename(file)

        if label not in dataset:
            dataset[label] = []
            os.mkdir(os.path.join(tmp_dir, label))

        if file_name in dataset[label] and not overwrite_duplicates:
            target_file_name = f"{int(time.time())}_{file_name}"
        else:
            target_file_name = file_name

        shutil.move(file, os.path.join(tmp_dir, label, target_file_name))
        dataset[label].append(target_file_name)

    print(f"Found and moved to {tmp_dir}:")
    print("\t" + "\n\t".join(f"[{k}] {len(v)} items" for k, v in dataset.items()) + "\n")

    print("Delete subdirectories? :\n\t" + "\n\t".join(sub_directories))
    input_key = input("Delete them: (y/n)")

    if input_key == "y":
        print("Will delete them.")
        for dd in sub_directories:
            shutil.rmtree(dd)

    # -- Split and move to folders
    split = np.array(split)
    split = split / np.sum(split)
    for split_name in split_names:
        split_folder = os.path.join(base_path, split_name)
        if os.path.isdir(split_folder):
            input_key = input(f"Folder {split_folder} exists!\n\tDelete it: (y/n)")
            assert input_key == "y", "Split Folder exists. Remove and try again."
        os.mkdir(split_folder)

    dataset_splits = {k: {} for k in split_names}

    for label, file_names in dataset.items():
        random.shuffle(file_names)
        label_cnt = (split * len(file_names)).astype(np.int)
        label_cnt[-1] = len(file_names) - label_cnt.sum()
        split_idx = np.cumsum(label_cnt)[:-1]
        file_splits = np.split(file_names, split_idx)
        for split_name, file_name in zip(split_names, file_splits):
            dataset_splits[split_name][label] = file_name

            labal_dir = os.path.join(base_path, split_name, label)
            os.mkdir(labal_dir)
            for file in file_name:
                shutil.move(os.path.join(tmp_dir, label, file), os.path.join(labal_dir, file))

    print("Splits:")
    for split in split_names:
        print(f"\t{split}")
        print("\t\t" +
              "\n\t\t".join(f"[{k}] {len(v)} items" for k, v in dataset_splits[split].items()))

    shutil.rmtree(tmp_dir)
    print("Done")


