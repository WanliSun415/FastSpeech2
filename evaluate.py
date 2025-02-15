import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
import wandb
from utils.model import vocoder_infer


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device('cuda:0')
print('Using device: ', device)


def evaluate(model, step, configs, logger=None, vocoder=None, infer_step=0):
    preprocess_config, model_config, train_config = configs
    vocoder = get_vocoder(model_config, device)
    # Get dataset
    dataset = Dataset(
        train_config["path"]['val_dataset'], preprocess_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Evaluation
    loss_sums = [0 for _ in range(9)]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            # with torch.no_grad():
                # Forward
            output = model(*(batch[2:]))

            # Cal Loss
            losses = Loss(batch, output)


            for i in range(len(losses)):
                loss_sums[i] += losses[i].item() * len(batch[0])
    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

    message = ("Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, "
               "Energy Loss: {:.4f}, Duration Loss: {:.4f}, SSM1 Loss: {:.4f}, SSM2 Loss: {:.4f}, Delta: {:.4f}").format(
        *([step] + [l for l in loss_means])
    )

    # do iterative inference
    if (not model.training) and (infer_step > 0):
        mel_len = output[9][0].item()
        steps_prediction = output[1][0, :mel_len].detach().transpose(0, 1)
        logits = output[12][0, :mel_len].detach().transpose(0, 1)
        for i in range(step):
            steps_prediction = steps_prediction + logits
        wav_step_prediction = vocoder_infer(
            steps_prediction.unsqueeze(0),
            vocoder,
            model_config,
            preprocess_config,
        )[0]
    # if logger is not None:
    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
        batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
        )

        # log(logger, step, losses=loss_means)
        # log(
        #     logger,
        #     fig=fig,
        #     tag="Validation/step_{}_{}".format(step, tag),
        # )
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        # log(
        #     logger,
        #     audio=wav_reconstruction,
        #     sampling_rate=sampling_rate,
        #     tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        # )
        # log(
        #     logger,
        #     audio=wav_prediction,
        #     sampling_rate=sampling_rate,
        #     tag="Validation/step_{}_{}_synthesized".format(step, tag),
        # )
    wandb.log({
        "val_epoch/total_loss": losses[0].item(),
        "val_epoch/mel_loss": losses[1].item(),
        "val_epoch/post_mel_loss": losses[2].item(),
        "val_epoch/pitch_loss": losses[3].item(),
        "val_epoch/energy_loss": losses[4].item(),
        "val_epoch/duration_loss": losses[5].item(),
        "val_epoch/ssm_loss1": losses[6].item(),
        "val_epoch/ssm_loss2": losses[7].item(),
        "val_epoch/mel_spectrogram": fig,
        "val_epoch/mel_to_vocoder_wav": wandb.Audio(wav_reconstruction, caption=batch[0][0], sample_rate=sampling_rate),
        "val_epoch/post_net_wav": wandb.Audio(wav_prediction, caption=batch[0][0], sample_rate=sampling_rate),
        "val_epoch/iter_wav": wandb.Audio(wav_step_prediction, caption=batch[0][0], sample_rate=sampling_rate)})
    return message


if __name__ == "__main__":
    wandb.init(project="evaluation")
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=30000)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
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
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False).to(device)

    message = evaluate(model, args.restore_step, configs, infer_step=10)
    print(message)