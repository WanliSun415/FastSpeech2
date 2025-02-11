import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths
from unet_1dconv import Decoder as Score
import torch.autograd as autograd
import numpy as np


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        self.score_net = Score(preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                               preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        pos_mel=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):

        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)



        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)


        postnet_output = self.postnet(output) + output

        postnet_output.requires_grad_(True)
        # Batch_size * Length * 80
        logits = self.score_net(postnet_output, pos_mel)

        # B * L * 80
        # vectors = t.randn_like(mel_input)
        vectors = torch.randn_like(torch.zeros(postnet_output.shape)).to(postnet_output.device)

        # score, a.k.a. gradient of logP, negtive gradient of energy
        grad1 = logits

        # mel masking, e.g. shape: B*L
        mel_mask = pos_mel.ne(0).type(torch.float)
        # length of mel, e.g. shape: B,
        mel_length = mel_mask.sum(dim=-1)
        # shape: B * L * 1
        mel_mask = mel_mask.view(pos_mel.shape[0], -1, 1)

        # shape: B,
        gradv = torch.sum(torch.sum(grad1 * vectors * mel_mask, dim=-1) / 80, dim=-1) / mel_length

        # second term in Eq. 8, shape: B,
        loss2 = torch.sum(torch.sum(grad1 * grad1 * mel_mask, dim=-1) / 80, dim=-1) / 2 / mel_length

        grad2 = autograd.grad(gradv.mean(), postnet_output, create_graph=True)[0]

        # first term in Eq. 8, shape: B *
        loss1 = torch.sum(torch.sum(vectors * grad2 * mel_mask, dim=-1) / 80, dim=-1) / mel_length

        loss1 = loss1.mean()
        loss2 = loss2.mean()

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            loss1,
            loss2
        )