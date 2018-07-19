import json
import os
import random

def split(annotations, *param, random_shuffle=True):
    train_ann = {}
    val_ann = {}
    test_ann = {}

    if random_shuffle == True:
        index = list(range(len(annotations)))
        random.shuffle(index)
        num_train = round(len(annotations)*0.6)
        num_val = round(len(annotations)*0.2)
        train_index = index[:num_train]
        val_index = index[num_train:num_train+num_val]
        test_index = index[num_train+num_val:]
        for each in train_index:
            key = list(annotations.keys())[each]
            train_ann[key] = annotations[key]
        for each in val_index:
            key = list(annotations.keys())[each]
            val_ann[key] = annotations[key]
        for each in test_index:
            key = list(annotations.keys())[each]
            test_ann[key] = annotations[key]
        return train_ann, val_ann, test_ann
    else:
        keys = list(annotations.keys())
        train_names = param[0]["train"]
        val_names = param[0]["val"]

        for key in keys:
            filename = annotations[key]["filename"]
            if filename in train_names:
                train_ann[key] = annotations[key]
            elif filename in val_names:
                val_ann[key] = annotations[key]
            else:
                test_ann[key] = annotations[key]
        return train_ann, val_ann, test_ann

if __name__ == "__main__":
    dataset_dir = "/home/simon/deeplearning/git_surgery/data/surgery/train"
    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    # if you want to specify train, val, test set with a dict, else just leave split_map as a empty dic
    split_map = {"train":["Picture 390.jpg"], "val":["Picture 392.jpg"], "test":[]}
    if len(split_map) == 0:
        train_ann, val_ann, test_ann = split(annotations)
    else:
        train_ann, val_ann, test_ann = split(annotations, split_map, random_shuffle=False)