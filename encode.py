import argparse

import torch.nn
import torchvision
from compressai.zoo import bmshj2018_factorized, cheng2020_anchor
from diffusers import StableDiffusion3Img2ImgPipeline, StableDiffusionImg2ImgPipeline
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.encoder_decoder import FED, INL
from models.vae import VAE
from utils.datasets import EncodeDataset
from utils.jpeg import JpegTest
from utils.metric import *
from utils.utils import *


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='ENCODE')
    parser.add_argument(
        '--noise-type',
        '-n',
        default='HEAVY',
        type=str,
        help='The noise type will be added to the watermarked images.',
    )
    parser.add_argument('--batch-size', '-b', default=1, type=int, help='The batch size.')
    parser.add_argument(
        '--source-image', '-s', default="test_images", type=str, help='The images to watermark'
    )
    parser.add_argument(
        '--source-image-type', '-t', default="png", type=str, help='The type of the input images'
    )
    parser.add_argument(
        '--messages-path', '-m', default="messages", type=str, help='The messages to embed'
    )
    parser.add_argument(
        '--watermarked-image', '-o', default="output_images", type=str, help='The output images'
    )
    parser.add_argument('--strength', default=0.1, type=float, help='sd img2ge strength')

    args = parser.parse_args()

    inn_data = EncodeDataset(args.source_image, args.source_image_type)
    inn_loader = DataLoader(inn_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    psnr_history1 = []
    psnr_history2 = []

    sd3 = StableDiffusion3Img2ImgPipeline.from_pretrained(
        'stabilityai/stable-diffusion-3.5-medium', torch_dtype=torch.bfloat16
    )
    sd3.enable_model_cpu_offload()

    # sd2 = StableDiffusionImg2ImgPipeline.from_pretrained(
    #     'stabilityai/stable-diffusion-2-1', torch_dtype=torch.float16
    # )
    # sd2.enable_model_cpu_offload()

    # vae2 = VAE()
    # vae2.model = sd2.vae.to(torch.float32)

    # vaeB = bmshj2018_factorized(quality=3, pretrained=True).eval().to(device)
    # vaeC = cheng2020_anchor(quality=3, pretrained=True).eval().to(device)

    with torch.inference_mode():
        if args.noise_type in ["JPEG", "HEAVY"]:
            if args.noise_type == "JPEG":
                noise_layer = JpegTest(50)
            vae = VAE()
            fed_path = os.path.join(
                'experiments/1218_21:39:51/JPEGfed_415_0.92810_99.95117%_99.98372%.pt'
            )
            fed = FED(vae.latent_channels).to(device)
            load(fed_path, fed)
            fed.eval()
            for idx, img1 in enumerate(tqdm(inn_loader)):
                img1: Tensor = img1.to(device)
                x1 = vae.encode(img1)
                source_messgaes1 = torch.Tensor(
                    np.random.choice([-0.5, 0.5], (x1.shape[0], 64))
                ).to(device)
                key1 = (
                    torch.randint(0, 2, source_messgaes1.shape, dtype=torch.int8).to(device) * 2
                    - 1
                )

                stego_x1, *_ = fed([x1, source_messgaes1, key1])
                stego_images1 = vae.decode(stego_x1)

                # img2 = stego_images1

                # img2 = vae(stego_images1)
                # img2 = vae2(stego_images1)
                # img2 = vaeB(stego_images1 * 0.5 + 0.5)['x_hat'] * 2 - 1
                # img2 = vaeC(stego_images1 * 0.5 + 0.5)['x_hat'] * 2 - 1

                img2 = sd3(
                    '',
                    image=(stego_images1 * 0.5 + 0.5).clamp(0, 1),
                    strength=args.strength,
                    num_inference_steps=40,
                    guidance_scale=4.5,
                    output_type='pt',
                ).images
                # img2 = sd2('', (stego_images1 * 0.5 + 0.5).clamp(0, 1)).images
                img2 = img2 * 2 - 1
                img2 = img2.to(torch.float32)

                x2 = vae.encode(img2)
                source_messgaes2 = torch.Tensor(
                    np.random.choice([-0.5, 0.5], (x2.shape[0], 64))
                ).to(device)
                key2 = (
                    torch.randint(0, 2, source_messgaes2.shape, dtype=torch.int8).to(device) * 2
                    - 1
                )
                stego_x2, *_ = fed([x2, source_messgaes2, key2])
                stego_images2 = vae.decode(stego_x2)

                psnr_value1 = psnr(img1, stego_images1, 255)
                psnr_value2 = psnr(img2, stego_images2, 255)

                for i in range(img1.shape[0]):
                    number = 1 + i + idx * img1.shape[0]
                    torchvision.utils.save_image(
                        ((stego_images1[i] / 2) + 0.5),
                        os.path.join(args.watermarked_image, "{}_1W.png".format(number)),
                    )
                    torchvision.utils.save_image(
                        ((img2[i] / 2) + 0.5),
                        os.path.join(args.watermarked_image, "{}_2.png".format(number)),
                    )
                    torchvision.utils.save_image(
                        ((stego_images2[i] / 2) + 0.5),
                        os.path.join(args.watermarked_image, "{}_2W.png".format(number)),
                    )

                torch.save(
                    source_messgaes1,
                    os.path.join(args.messages_path, "message_{}_1.pt".format(idx + 1)),
                )
                torch.save(
                    source_messgaes2,
                    os.path.join(args.messages_path, "message_{}_2.pt".format(idx + 1)),
                )
                torch.save(
                    key1,
                    os.path.join(args.messages_path, "key_{}_1.pt".format(idx + 1)),
                )
                torch.save(
                    key2,
                    os.path.join(args.messages_path, "key_{}_2.pt".format(idx + 1)),
                )

                psnr_history1.append(psnr_value1)
                psnr_history2.append(psnr_value2)

        else:
            raise ValueError("\"{}\" is not a valid noise type ".format(args.noise_type))

    print('psnr1: {:.3f}'.format(np.mean(psnr_history1)))
    print('psnr2: {:.3f}'.format(np.mean(psnr_history2)))

    with open('test_strengths.log', 'a') as f:
        f.write('-' * 50 + '\n')
        f.write(f'{args.strength=}\n')
        f.write(f'psnr1: {np.mean(psnr_history1):.3f}\n')
        f.write(f'psnr2: {np.mean(psnr_history2):.3f}\n')


if __name__ == '__main__':
    main()
