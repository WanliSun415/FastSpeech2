import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss, ContrastiveRegressionLoss
from dataset import Dataset

from evaluate import evaluate, validate
from save_and_plot import load_epoch_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, fake_preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    fake_dataset = Dataset(
        "train.txt", fake_preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=fake_dataset.collate_fn,
    )
    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    # model = nn.DataParallel(model)
    num_param = get_param_num(model)
    # Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    Loss = ContrastiveRegressionLoss()
    print("Number of FastSpeech2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    with open(os.path.join(train_config['path']['result_path'], "train_loss.json"), 'w') as fp:
        pass
    with open(os.path.join(train_config['path']['result_path'], "valid_loss.json"), 'w') as fp:
        pass

    pos_energy = []
    neg_energy = []
    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        epoch_loss = 0
        pos_epoch_loss = 0
        neg_epoch_loss = 0
        train_loss_json = {}
        val_loss_json = {}

        for batchs, fake_batchs in zip(loader, fake_loader):
            for batch, fake_batch in zip(batchs, fake_batchs):
                batch = to_device(batch, device)
                fake_batch = to_device(fake_batch, device)
                # Forward
                output = model(*(batch))
                fake_output = model(*(fake_batch))
                # Cal Loss
                # losses = Loss(batch, output)
                # total_loss = losses[0]
                pos_energy += output[0].data.cpu().tolist()
                neg_energy += fake_output[0].data.cpu().tolist()
                pos_loss = Loss(-output[0])
                neg_loss = Loss(fake_output[0])
                total_loss = pos_loss + neg_loss
                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                epoch_loss += total_loss.item() / 2
                pos_epoch_loss += pos_loss.item()
                neg_epoch_loss += neg_loss.item()

                # if step % log_step == 0:
                #     losses = [l.item() for l in losses]
                #     message1 = "Step {}/{}, ".format(step, total_step)
                #     message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                #         *losses
                #     )

                    # with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    #     f.write(message1 + message2 + "\n")
                    #
                    # outer_bar.write(message1 + message2)
                    #
                    # log(train_logger, step, losses=losses)

                # if step % synth_step == 0:
                #     fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                #         batch,
                #         output,
                #         vocoder,
                #         model_config,
                #         preprocess_config,
                #     )
                #     log(
                #         train_logger,
                #         fig=fig,
                #         tag="Training/step_{}_{}".format(step, tag),
                #     )
                #     sampling_rate = preprocess_config["preprocessing"]["audio"][
                #         "sampling_rate"
                #     ]
                #     log(
                #         train_logger,
                #         audio=wav_reconstruction,
                #         sampling_rate=sampling_rate,
                #         tag="Training/step_{}_{}_reconstructed".format(step, tag),
                #     )
                #     log(
                #         train_logger,
                #         audio=wav_prediction,
                #         sampling_rate=sampling_rate,
                #         tag="Training/step_{}_{}_synthesized".format(step, tag),
                #     )


                # if step % val_step == 0:
                #     model.eval()
                #     message = evaluate(model, step, configs, val_logger, vocoder)
                #     with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                #         f.write(message + "\n")
                #     outer_bar.write(message)
                #
                #     model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(train_config["path"]["ckpt_path"], "{}.pth.tar".format(step),),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)

        train_loss_json[str(epoch)] = {"epoch_loss": epoch_loss / len(dataset),
                                       "pos_epoch_loss": pos_epoch_loss / len(dataset),
                                       "neg_epoch_loss": neg_epoch_loss / len(dataset),
                                       "pos_energy": pos_energy[-512:],
                                       "neg_energy": neg_energy[-512:]}
        pos_energy = []
        neg_energy = []
        val_epoch_loss, val_pos_energy, val_neg_energy = validate(model, configs, epoch)
        val_loss_json[str(epoch)] = {"epoch_loss": val_epoch_loss[2],
                                     "pos_epoch_loss": val_epoch_loss[0],
                                     "neg_epoch_loss": val_epoch_loss[1],
                                     "pos_energy": val_pos_energy,
                                     "neg_energy": val_neg_energy}

        load_epoch_loss(train_loss_json, os.path.join(train_config['path']['result_path'], "train_loss.json"))
        load_epoch_loss(val_loss_json, os.path.join(train_config['path']['result_path'], "valid_loss.json"))
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-fp",
        "--fake_preprocess_config",
        type=str,
        required=True,
        help="path to fake_preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    fake_preprocess_config = yaml.load(
        open(args.fake_preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, fake_preprocess_config, model_config, train_config)

    main(args, configs)
