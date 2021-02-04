import scipy.misc
import numpy as np
import torch
from vae import VAE
from argparse import ArgumentParser


def main(args):
    dim = 64
    latent_size = 512
    channels = 3
    best_model_path = "./vae_lambda001.pth"
    # should give from outside
    output_path = args.output_path

    # load model
    model = VAE(d=dim, zsize=latent_size, channels=channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.load_state_dict(torch.load(
        best_model_path, map_location=device))

    # 4. sample from normal distribution
    seed = 5
    np.random.seed(seed)
    samples = np.random.randn(32, latent_size)
    torch_samples = torch.FloatTensor(samples)
    torch_samples = torch_samples.to(device)
    result = model.decode(torch_samples)

    # concat images
    rows = []
    for row in range(8):
        rows.append(
            torch.cat([img for img in result[row * 4: (row+1) * 4]], axis=1))

    ret = torch.cat(rows, axis=2)

    img = ret.cpu().detach().numpy().transpose(1, 2, 0)
    scipy.misc.imsave(output_path, img)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--output_path", help="path of the test image directory",
                        dest="output_path", default="./images/vae.png")

    args = parser.parse_args()
    main(args)
