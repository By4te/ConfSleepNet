import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd

# from sleepstage import stage_dict
from logger import get_logger

"""AASM Sleep Manual Label"""

# Label values
W = 0  # Stage AWAKE
N1 = 1  # Stage N1
N2 = 2  # Stage N2
N3 = 3  # Stage N3
REM = 4  # Stage REM
MOVE = 5  # Movement
UNK = 6  # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}

class_dict = {
    W: "W",
    N1: "N1",
    N2: "N2",
    N3: "N3",
    REM: "REM",
    MOVE: "MOVE",
    UNK: "UNK",
}

# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,  # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="G:/Data/SleepEdf39/origin",
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default="G:/Data/SleepEdf39/pre_processing",
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default="EEG Fpz-Cz",
                        help="Name of the channel in the dataset.")
    parser.add_argument("--log_file", type=str, default="info_ch_extract.log",
                        help="Log file.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    args.log_file = os.path.join(args.output_dir, args.log_file)

    # Create logger
    logger = get_logger(args.log_file, level="info")

    # Select channel
    select_ch = args.select_ch

    # Read raw and annotation from EDF files
    psg_fnames = glob.glob(os.path.join(args.data_dir, "*PSG.edf"))
    ann_fnames = glob.glob(os.path.join(args.data_dir, "*Hypnogram.edf"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):
        # for i in range(1):

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = psg_f.datarecord_duration
        if psg_f.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()  # EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental
        ch_samples = psg_f.getNSamples()  # 7950000,    7950000,   7950000,        79500

        # ************************ 1. EEG Fpz-Cz ************************
        select_ch_idx1 = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == 'EEG Fpz-Cz':
                select_ch_idx1 = s
                break
        if select_ch_idx1 == -1:
            raise Exception("Channel not found.")
        sampling_rate1 = psg_f.getSampleFrequency(select_ch_idx1)
        n_epoch_samples1 = int(epoch_duration * sampling_rate1)
        signals1 = psg_f.readSignal(select_ch_idx1).reshape(-1, n_epoch_samples1)  # shape: (2650, 3000)
        logger.info("Select channel: Fpz-Cz")
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx1]))
        logger.info("Sample rate: {}".format(sampling_rate1))

        # ************************ 2. EEG Pz-Oz ************************
        select_ch_idx2 = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == 'EEG Pz-Oz':
                select_ch_idx2 = s
                break
        if select_ch_idx2 == -1:
            raise Exception("Channel not found.")
        sampling_rate2 = psg_f.getSampleFrequency(select_ch_idx2)
        n_epoch_samples2 = int(epoch_duration * sampling_rate2)
        signals2 = psg_f.readSignal(select_ch_idx2).reshape(-1, n_epoch_samples2)  # shape: (2650, 3000)

        # ************************ 3. EOG horizontal ************************
        select_ch_idx3 = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == 'EOG horizontal':
                select_ch_idx3 = s
                break
        if select_ch_idx3 == -1:
            raise Exception("Channel not found.")
        sampling_rate3 = psg_f.getSampleFrequency(select_ch_idx3)
        n_epoch_samples3 = int(epoch_duration * sampling_rate3)
        signals3 = psg_f.readSignal(select_ch_idx3).reshape(-1, n_epoch_samples3)  # shape: (2650, 3000)

        # Sanity check
        n_epochs = psg_f.datarecords_in_file
        if psg_f.datarecord_duration == 60:  # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert len(signals1) == n_epochs, f"signal1: {signals1.shape} != {n_epochs}"
        assert len(signals2) == n_epochs, f"signal2: {signals2.shape} != {n_epochs}"
        assert len(signals3) == n_epochs, f"signal3: {signals3.shape} != {n_epochs}"

        # ************************ Generate labels *************************
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()  # the three are of shape: (154, )
        # print('ann_stages:', ann_stages.shape)

        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)

        # Remove annotations that are longer than the recorded signals
        labels = labels[:len(signals1)]

        # Get epochs and their corresponding labels
        x1 = signals1.astype(np.float32)
        x2 = signals2.astype(np.float32)
        x3 = signals3.astype(np.float32)
        y = labels.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        print('y:', y.shape)
        nw_idx = np.where(y != stage_dict["W"])[0]

        start_idx = nw_idx[0] - (w_edge_mins * 2)
        end_idx = nw_idx[-1] + (w_edge_mins * 2)
        if start_idx < 0: start_idx = 0
        if end_idx >= len(y): end_idx = len(y) - 1
        select_idx = np.arange(start_idx, end_idx + 1)
        x1 = x1[select_idx]
        x2 = x2[select_idx]
        x3 = x3[select_idx]
        y = y[select_idx]

        # Remove movement and unknown
        move_idx = np.where(y == stage_dict["MOVE"])[0]
        unk_idx = np.where(y == stage_dict["UNK"])[0]
        if len(move_idx) > 0 or len(unk_idx) > 0:
            remove_idx = np.union1d(move_idx, unk_idx)
            select_idx = np.setdiff1d(np.arange(len(x1)), remove_idx)
            x1 = x1[select_idx]
            x2 = x2[select_idx]
            x3 = x3[select_idx]
            y = y[select_idx]
            assert len(x1) == len(x2)
            assert len(x2) == len(x3)
            assert len(x3) == len(y)
            n_epochs = len(x3)

        print('x1-3:', x1.shape, x2.shape, x3.shape, y.shape)

        # Save
        filename = ntpath.basename(psg_fnames[i]).replace("-PSG.edf", ".npz")
        save_dict = {
            "EEG Fpz-Cz": x1,
            "EEG Pz-Oz": x2,
            "EOG": x3,
            "y": y,
            "fs": sampling_rate1,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_epochs": n_epochs,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)

        logger.info("\n=======================================\n")


if __name__ == "__main__":
    main()

