import soundfile as sf
import argparse
import torch
import os
import pandas as pd
import torchaudio
import models.exp_models as M


SAMPLING_RATE = 16000


# def run_train(spk_id, size, learning_rate, model_name, partition='120sec', load_ckpt=False, max_iter=None,
#               min_epoch=0, max_epoch=200, is_rand_val=False, sampler_name=None, save_folder=None, batch_size=None):
def run_inference(spk_id, learning_rate, size, partition, checkpoint_path, test_data_csv_dir, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare paths
    run_name = f"spk{spk_id}_{size}_{learning_rate}_{partition}"
    checkpoint_path = os.path.join(checkpoint_path, run_name, "model_best.ckpt")
    assert os.path.exists(checkpoint_path), f"checkpoint_path {checkpoint_path} does not exist"
    csv_path = os.path.join(test_data_csv_dir, f"{partition}.csv")

    print(f"> run name: {run_name}")
    print(f"> speaker: {spk_id}")

    # Load model
    net, nparams, config = M.init_model("convtasnet", size)
    net = net.to(device)
    net.load_state_dict(torch.load(checkpoint_path).get("model_state_dict"), strict=True)
    net.eval()

    # Read csv file and prepare test data
    data = pd.read_csv(csv_path)
    data = data[(data["spk"] == spk_id) & (data["split"] == "test")]
    test_wav_files = sorted(data["file"].tolist())

    for i, x in enumerate(test_wav_files):
        # Load data
        x, fs = torchaudio.load(x)
        if fs != SAMPLING_RATE:
            x = torchaudio.functional.resample(x, fs, SAMPLING_RATE)
        x = x.to(device)

        # Inference
        y = M.make_2d(net(x.unsqueeze(0)))
        save_wav_path = f"{save_path}/{spk_id}/{spk_id}_task2_{i:02d}.wav"
        if not os.path.exists(os.path.dirname(save_wav_path)):
            os.makedirs(os.path.dirname(save_wav_path))
        sf.write(save_wav_path, y.squeeze(0).detach().cpu().numpy(), SAMPLING_RATE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--speaker_id", type=str, required=True)
    parser.add_argument("-r", "--learning_rate", type=float, required=True)
    parser.add_argument("-i", "--size", type=str, required=True)
    parser.add_argument("-p", "--partition", type=str, required=True)
    # parser.add_argument("-m", "--model_name", type=str, required=True)
    # parser.add_argument("-ex", "--experiment", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="results/test")
    parser.add_argument("--test_data_csv_dir", type=str, default="examples/csv_files")
    parser.add_argument("--save_path", type=str, default="examples/enhanced_wavs")

    args = parser.parse_args()

    run_inference(
        spk_id=args.speaker_id,
        learning_rate=args.learning_rate,
        size=args.size,
        partition=args.partition,
        checkpoint_path=args.checkpoint_path,
        test_data_csv_dir=args.test_data_csv_dir,
        save_path=args.save_path,
    )
