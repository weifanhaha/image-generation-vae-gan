import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import scipy.misc

from models import Generator
from image_dataset import ImageDataset
from argparse import ArgumentParser


def main(args):

    nz = 100
    ngf = 64

    best_model_path = "./best_G.pth"
    output_path = args.output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator(nz=nz, ngf=ngf, nc=3)
    netG = netG.to(device)

    netG.load_state_dict(torch.load(
        best_model_path, map_location=device))

    seed = 0
    torch.manual_seed(seed)

    noise = torch.randn(32, nz, 1, 1, device=device)
    imgs = netG(noise)

    rows = []
    for row in range(8):
        rows.append(
            torch.cat([img for img in imgs[row * 4: (row+1) * 4]], axis=1))

    ret = torch.cat(rows, axis=2)

    img = ret.cpu().detach().numpy().transpose(1, 2, 0)
    scipy.misc.imsave(output_path, img)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--output_path", help="path of the test image directory",
                        dest="output_path", default="./gan.png")

    args = parser.parse_args()
    main(args)
