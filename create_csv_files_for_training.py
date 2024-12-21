import os
import csv
import argparse


def utt_50(spks, wav_dir, model, csv_save_dir, prefix=""):
    num_utt = 50
    val_utt = 10

    out = [["spk", "split", "file"]]
    # Train
    for spk in spks:
        for i in range(num_utt - val_utt):
            path = f"{wav_dir}/{spk}/{spk}_{model}_{i:03d}.wav"
            assert os.path.exists(path), path
            out.append([spk, "train", path])

    # Val
    for spk in spks:
        for i in range(num_utt - val_utt, num_utt):
            path = f"{wav_dir}/{spk}/{spk}_{model}_{i:03d}.wav"
            assert os.path.exists(path), path
            out.append([spk, "val", path])

    with open(f"{csv_save_dir}/{model}_synth_50utt{prefix}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--wav_dir", type=str, default="examples/synth_data")
    parser.add_argument("--csv_save_dir", type=str, default="examples/csv_files")
    args = parser.parse_args()

    spks = sorted([x for x in os.listdir(args.data_dir) if os.path.isdir(f"{args.data_dir}/{x}")])
    # spks = ["F0"]  # For test we only used a single speaker
    print("# Processing speaker: ", spks)
    # print("# WARNING: For test purpose, we only used a single speaker")
    os.makedirs(args.csv_save_dir, exist_ok=True)

    utt_50(spks, wav_dir=args.wav_dir, model="speecht5", csv_save_dir=args.csv_save_dir, prefix="")
    print("# Generate CSV files for training successfully!")
