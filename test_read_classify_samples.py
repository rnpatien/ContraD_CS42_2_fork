from argparse import ArgumentParser
from pathlib import Path
import os
import math
from glob import glob

import gin
import torch
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from datasets import get_dataset
from models.gan import get_architecture

from training.gan import setup
from imageio import imread

# import for gin binding
import penalty
import augment

from models.gan.base import LinearWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = ArgumentParser(description='Testing script: Random sampling from G')
    parser.add_argument('imgdir', type=str, help='Path to the directory of generated images')
    parser.add_argument('logdir', type=str,
                        help='Path to the logdir that contains the (best) checkpoints of G and D')
    parser.add_argument('linear_path', type=str,
                        help='Path to the checkpoint trained from linear evaluation')
    parser.add_argument('architecture', type=str, help='Architecture')

    parser.add_argument('--n_samples', default=2000, type=int,
                        help='Number of samples to generate (default: 10000)')
    parser.add_argument('--batch_size', default=500, type=int,
                        help='Batch size (default: 100)')

    return parser.parse_args()


# def _sample_generator(G, num_samples):
#     latent_samples = G.sample_latent(num_samples)
#     generated_data = G(latent_samples)
#     return generated_data

def _descrimator(D, samples):
    d_out, aux = D(samples, penultimate=True)
    penul = aux['penultimate']
    logit=D.classifier(penul)
    pred= torch.argmax(logit,axis=1)
    return pred


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

def init_logfile(filename: str, text: str):
    f = open(filename, 'w')
    f.write(text+"\n")
    f.close()


def fwrite(filename: str, text: str):
    f = open(filename, 'a')
    f.write(text+"\n")
    f.close()

import sys


if __name__ == '__main__':
    print ('Argument List:', str(sys.argv))
    P = parse_args()

    gin_config = sorted(glob(f"{P.logdir}/*.gin"))[0]
    gin.parse_config_files_and_bindings(['configs/defaults/gan.gin',
                                         'configs/defaults/augment.gin',
                                         gin_config], [])
    options = get_options_dict()
    n_classes=10

    _, _, image_size = get_dataset(dataset=options['dataset'])

    _, discriminator = get_architecture(P.architecture, image_size)
    _, discriminator_l = get_architecture(P.architecture, image_size)
    discriminator_l.linear = LinearWrapper(discriminator_l.d_penul, n_classes)
     
    checkpoint_d = torch.load(f"{P.logdir}/dis_best.pt")
    checkpoint_l = torch.load(f"{P.linear_path}")["state_dict"]
     
    discriminator.load_state_dict(checkpoint_d)
    discriminator_l.load_state_dict(checkpoint_l)
    discriminator.classifier = discriminator_l.linear
     
    discriminator.to(device).eval()

    root_dir=P.imgdir
    GBL_FN = str(root_dir +'/'+ f'class_distribution.csv') # max min
    init_logfile(GBL_FN, "Distribution list")

    theList=["DC_FL","DC_IM","DC_PR","CD_FL","CD_IM","CD_PR","DM_FL","DM_IM","DM_PR"]
    for exp in theList:
        NEW_PATH_DATA=root_dir + exp + "_Class/"
        if not os.path.exists(NEW_PATH_DATA):
            os.mkdir(NEW_PATH_DATA)
            for ii in range(10):
                    os.mkdir(NEW_PATH_DATA+ '/'+ str(ii)) 

    for exp in theList:

        PATH_DATA=root_dir + exp  +"/"
        NEW_PATH_DATA=root_dir + exp + "_Class/"
        
        data =  sorted(list(glob(PATH_DATA+'*.png')),key=lambda name: int(name.split('.')[0].split('/')[-1]))


        # subdir_path = P.logdir +'/'+ f"samples_{np.random.randint(10000)}_n{P.n_samples}"
        Xorg = np.array([imread(str(data[i])).astype(np.float32) for i in range(P.n_samples)])
        Xorg= (Xorg/255)
        X=np.transpose(Xorg,(0,3,1,2))
        print("done")
        print("# image values in range [%.2f, %.2f]" % (X.min(), X.max()))


        clsCnts=[0]*10
        n_batches = int(math.ceil(P.n_samples / P.batch_size))
        for i in tqdm(range(n_batches)):
            offset = i * P.batch_size
            samples= torch.Tensor(X[offset:offset+P.batch_size])
            samples=samples.cuda()
            with torch.no_grad():
                # samples = _sample_generator(generator, P.batch_size)
                pred= _descrimator(discriminator,samples)

                samples = samples.cpu()
                pred = pred.cpu()
            for j in range(samples.size(0)):
                index = offset + j
                if index == P.n_samples:
                    break
                prd=pred[j]
                clsCnts[prd] +=1
                save_image(samples[j], f"{NEW_PATH_DATA}/{prd}/{index}.png")

        print("class dirstribution =",clsCnts)
        stclscnts=[str(x) for x in clsCnts]
        fwrite(GBL_FN, exp + ' , '+  ','.join(stclscnts) )
    aa=1