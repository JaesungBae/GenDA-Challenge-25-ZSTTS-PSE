import argparse
import os
import time

# import asteroid
import models.exp_models as M
import torch
import wandb

# import pprint
from utils.dataloaders import SamplerFixNoise
from utils.exp_data import neg_sdr, sdr_improvement

# from asteroid.losses.sdr import singlesrc_neg_snr


GENERALIST_CHECKPOINT_DIR = "./checkpoints/generalists"
TRAIN_DATA_FILE_DIR = "./data/csv_files"
VALIDATION_EPOCH = 10  # Run validation every 10 epochs


def run_train(
    spk_id,
    size,
    learning_rate,
    partition,
    save_folder,
    train_data_csv_dir,
    musan_data_path,
    spk_noise_set_path,
    batch_size=None,
    min_epoch=0,
    max_epoch=200,
):
    """
    Args:
        spk_id: speaker id
        size: size of the model (e.g., tiny, small, medium)
        learning_rate: learning rate
        partition (str): name of dataset used for training. Same as training data csv file name.
        min_epoch: minimum epoch to run
        max_epoch: maximum epoch to run
        save_folder: save folder for the checkpoint
        batch_size: batch size
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model, optimizer, loss function
    net, nparams, config = M.init_model("convtasnet", size)
    net = net.to(device)
    loss_sdr = neg_sdr
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load pre-trained generalist checkpoint
    for f in os.listdir(GENERALIST_CHECKPOINT_DIR):
        if size in f:
            checkpoint_path = f"{GENERALIST_CHECKPOINT_DIR}/{f}"
    net.load_state_dict(torch.load(checkpoint_path).get("model_state_dict"), strict=True)
    optimizer.load_state_dict(torch.load(checkpoint_path).get("optimizer_state_dict"))

    # Refine learning rate
    for param_group in optimizer.param_groups:
        print(param_group["lr"], "changed to", learning_rate)
        param_group["lr"] = learning_rate
    # Load pre-trained generalist checkpoint end

    # Reset config
    config["batch_size"] = batch_size
    config["loss_type"] = "neg_sdr"
    config["spk_id"] = spk_id
    config["learning_rate"] = learning_rate
    config["partition"] = partition
    csv_path = os.path.join(train_data_csv_dir, f"{partition}.csv")

    # Set run name for checkpoint saving
    run_name = f"spk{spk_id}_{size}_{learning_rate}_{partition}"
    print(f"> run name: {run_name}")

    # Set save path
    save_path = os.path.join(save_folder, run_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch_val_loss = 0
    val_sdr_improvement = 0

    with wandb.init(config=config, project="pse"):
        wandb.run.name = run_name

        batch_sampler = SamplerFixNoise(
            csv_path, musan_data_path, spk_noise_set_path, mode="train", is_random_noise_cut=False
        )
        val_batch_sampler = SamplerFixNoise(
            csv_path, musan_data_path, spk_noise_set_path, mode="val", is_random_noise_cut=False
        )

        train_total_num = batch_sampler.get_data_len(spk_id, "train")
        print("# Num. of train samples:", train_total_num)

        # Load validation data for fixed val samples and noises.
        val_total_num = val_batch_sampler.get_data_len(spk_id, "val")  # repeat 5 times
        print("# Num. of original val samples:", val_total_num)
        print("# We will repeat 3 times for validation")
        val_batch_1 = val_batch_sampler.sample_batch(spk_id, val_total_num, "val")
        val_batch_2 = val_batch_sampler.sample_batch(spk_id, val_total_num, "val")
        val_batch_3 = val_batch_sampler.sample_batch(spk_id, val_total_num, "val")
        VAL_batch_size = 8

        val_x = torch.cat([val_batch_1["x"], val_batch_2["x"], val_batch_3["x"]], dim=0).to(device)
        val_t = torch.cat([val_batch_1["t"], val_batch_2["t"], val_batch_3["t"]], dim=0).to(device)
        val_total_num *= 3

        # Set hyperparameters for training
        wandb.watch(net, loss_sdr, log="all", log_freq=100)

        ep = 0
        prev_val_loss = 99999999
        no_improvement = 0
        total_steps = 0
        global_iter = 0

        # Set number of iterations for each batch
        seen_mixtures = 45  # Fixed batch size. so the ep is not actually epoch.
        if (seen_mixtures % config["batch_size"]) > 0:
            num_iter = (seen_mixtures // config["batch_size"]) + 1
        else:
            num_iter = seen_mixtures // config["batch_size"]

        for _ in range(max_epoch):
            # TRAIN
            s_time = time.time()
            epoch_train_loss = 0
            epoch_sdr_improvement = 0
            net.train()
            for i in range(num_iter):
                batch = batch_sampler.sample_batch(spk_id, config["batch_size"], "train")
                x = batch["x"].to(device)
                t = batch["t"].to(device)
                y = M.make_2d(net(x))
                loss = loss_sdr(y, t).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_train_loss += loss.data
                epoch_sdr_improvement += sdr_improvement(y, t, x, reduction="mean")
                total_steps += config["batch_size"]
                global_iter += 1

            epoch_train_loss /= num_iter
            epoch_sdr_improvement /= num_iter

            epoch_train_time = time.time() - s_time

            # VALIDATION
            if ep % VALIDATION_EPOCH == 0:
                s_time = time.time()
                epoch_val_loss = 0
                net.eval()

                mini_steps = val_x.shape[0] // VAL_batch_size
                for mini_batch_idx in range(mini_steps):
                    start = mini_batch_idx * VAL_batch_size
                    end = min(start + VAL_batch_size, val_x.shape[0])
                    x_mini = val_x[start:end]
                    t_mini = val_t[start:end]
                    y_mini = M.make_2d(net(x_mini)).detach()
                    loss = loss_sdr(y_mini, t_mini).mean().detach()
                    val_sdr_improvement += sdr_improvement(y_mini, t_mini, x_mini, reduction="mean")
                    epoch_val_loss += loss.data

                epoch_val_loss /= mini_steps
                val_sdr_improvement /= mini_steps
                epoch_val_time = time.time() - s_time
                wandb.log(
                    {
                        "train_loss": epoch_train_loss.data,
                        "val_loss": epoch_val_loss.data,
                        "train_sdr_improvement": epoch_sdr_improvement,
                        "val_sdr_improvement": val_sdr_improvement,
                    },
                    step=ep,
                )

                print(f"> [Epoch]:{ep+1} [Train Loss]: {epoch_train_loss:.4f}, takes {epoch_train_time:.2f} sec")
                print(f"> [Epoch]:{ep+1} [Val Loss]: {float(epoch_val_loss):.4f}, takes {epoch_val_time:.2f} sec")

                if epoch_val_loss < prev_val_loss:
                    prev_val_loss = epoch_val_loss

                    # Keep the best checkpoint
                    print("# Better performance in validation. Keep this for the best ckpt")
                    best_ckpt_name = f"{save_path}/model_best.ckpt"
                    best_ckpt = {}
                    best_ckpt["model_state_dict"] = net.state_dict()
                    best_ckpt["optim_state_dict"] = optimizer.state_dict()
                    best_ckpt["epoch"] = ep
                    best_ckpt["config"] = config
                    no_improvement = 0
                else:
                    no_improvement += 1

                if no_improvement >= 3 and ep > min_epoch:
                    print(f"> Training finished [epochs]:{ep+1} [steps]: {total_steps}")
                    # Save best checkpoint first
                    torch.save(best_ckpt, best_ckpt_name)
                    print("# Best checkpoint saved at, ", best_ckpt_name)

                    # Then save last checkpoint
                    ckpt_name = f"{save_path}/model_last_{ep+1}.ckpt"
                    ckpt = {}
                    ckpt["model_state_dict"] = net.state_dict()
                    ckpt["optim_state_dict"] = optimizer.state_dict()
                    ckpt["epoch"] = ep
                    ckpt["config"] = config
                    torch.save(ckpt, ckpt_name)
                    print("# Last checkpoint saved at, ", ckpt_name)
                    break
            ep += 1
        # Save best checkpoint before leaving
        torch.save(best_ckpt, best_ckpt_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--speaker_id", type=str, required=True)
    parser.add_argument("-r", "--learning_rate", type=float, required=True)
    parser.add_argument("-i", "--size", type=str, required=True)
    parser.add_argument("-p", "--partition", type=str, required=True)
    parser.add_argument("--min_epoch", type=int, default=0)
    parser.add_argument("--max_epoch", type=int, default=1001)
    parser.add_argument("--batch_size", type=int, default=8)

    # path settings
    parser.add_argument("--save_path", type=str, default="results/test")
    parser.add_argument("--train_data_csv_dir", type=str, default="examples/csv_files")
    parser.add_argument("--musan_data_path", type=str, default="data/musan/noise/sound-bible")
    parser.add_argument("--spk_noise_set_path", type=str, default="data/spk_noise_set.json")
    args = parser.parse_args()

    torch.cuda.empty_cache()

    print("########### Experiment Settings ##############")
    print("# save_path:", args.save_path)
    print("# Batch Size:", args.batch_size)
    print("##############################################")

    run_train(
        spk_id=args.speaker_id,
        size=args.size,
        learning_rate=args.learning_rate,
        partition=args.partition,
        save_folder=args.save_path,
        train_data_csv_dir=args.train_data_csv_dir,
        musan_data_path=args.musan_data_path,
        spk_noise_set_path=args.spk_noise_set_path,
        min_epoch=args.min_epoch,
        max_epoch=args.max_epoch,
        batch_size=args.batch_size,
    )
