from argparse import ArgumentParser
from pathlib import Path
import os
import math

import gin
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from models.gan import get_architecture

from training.gan import setup

# import for gin binding
import penalty
import augment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Testing script: Random sampling from G')
    parser.add_argument('model_path', type=str, help='Path to contraD')

    parser.add_argument('--n_samples', default=10000, type=int,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--batch_size', default=500, type=int,
                        help='Batch size (default: 500)')

    return parser.parse_args()


def _sample_generator(G, num_samples):
    latent_samples = G.sample_latent(num_samples)
    generated_data = G(latent_samples)
    return generated_data


@gin.configurable("options")
def get_options_dict(dataset=gin.REQUIRED,
                     loss=gin.REQUIRED,
                     batch_size=64, fid_size=10000,
                     max_steps=200000,
                     warmup=0,
                     n_critic=1,
                     lr=2e-4, lr_d=None, beta=(.5, .999),
                     lbd=10., lbd2=10.):
    if lr_d is None:
        lr_d = lr
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "fid_size": fid_size,
        "loss": loss,
        "max_steps": max_steps, "warmup": warmup,
        "n_critic": n_critic,
        "lr": lr, "lr_d": lr_d, "beta": beta,
        "lbd": lbd, "lbd2": lbd2
    }

import sys
if __name__ == '__main__':
    print ('Argument List:', str(sys.argv))
    P = parse_args()

    experiments=["DC_FL","DC_IM","DC_PR","CD_FL","CD_IM","CD_PR","DM_FL","DM_IM","DM_PR"]
    architect= {'DC':'sndcgan','CD':'snresnet18','DM':'snresPrune'}
    srcdics=["c10_b64/sndcgan/std_none/DC_NA_FL","c10_b64/sndcgan/std_none/DC_NA_IM","c10_b64/sndcgan/std_none/DC_NA_PR",
    "c10_b512/snresnet18/contrad_simclr_L1.0_T0.1/RS_CD_FL","c10_b512/snresnet18/contrad_simclr_L1.0_T0.1/RS_CD_IM","c10_b512/snresnet18/contrad_simclr_L1.0_T0.1/RS_CD_PR",
    "c10_b512/snresPrune/damage_simclr_L1.0_T0.1/RS_DM_FL","c10_b512/snresPrune/damage_simclr_L1.0_T0.1/RS_DM_IM","c10_b512/snresPrune/damage_simclr_L1.0_T0.1/RS_DM_PR"]


    tgtDics = "/mnt/e/5704_testcase/"
    if not os.path.exists(tgtDics):
        os.mkdir(tgtDics)
        for ii,exp in enumerate(experiments):
            if not  os.path.exists(tgtDics + exp):
                os.mkdir(tgtDics + exp)


    srcdir=Path(P.model_path + srcdics[0])
    gin_config = sorted(srcdir.glob("*.gin"))[0]
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         gin_config], [])
    options = get_options_dict()


    _, _, image_size = get_dataset(dataset=options['dataset'])

    for ii,exp in enumerate(experiments):
        arch=exp.split('_')[0]
        generator, _ = get_architecture(architect[arch], image_size)
        checkpoint = torch.load(P.model_path + srcdics[ii] + '/gen_best.pt')
        generator.load_state_dict(checkpoint)
        generator.to(device)
        generator.eval()

        print("Sampling in %s" % P.model_path + srcdics[ii])

        n_batches = int(math.ceil(P.n_samples / P.batch_size))
        for i in tqdm(range(n_batches)):
            offset = i * P.batch_size
            with torch.no_grad():
                samples = _sample_generator(generator, P.batch_size)
                samples = samples.cpu()
            for j in range(samples.size(0)):
                index = offset + j
                if index == P.n_samples:
                    break
                save_image(samples[j], f"{tgtDics + exp}/{index}.png")


