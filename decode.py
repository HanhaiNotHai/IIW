import argparse

import torch.nn
from torch.utils.data import DataLoader

from models.encoder_decoder import FED, INL
from utils.datasets import Test_Dataset
from utils.metric import *
from utils.utils import *


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser(description='DECODE')
    parser.add_argument(
        '--noise-type',
        '-n',
        default='JPEG',
        type=str,
        help='The noise type added to the watermarked images.',
    )
    parser.add_argument('--batch-size', '-b', default=16, type=int, help='The batch size.')
    parser.add_argument(
        '--messages-path', '-m', default="messages", type=str, help='The embedded messages'
    )
    parser.add_argument(
        '--watermarked-image',
        '-o',
        default="output_images",
        type=str,
        help='The watermarked images',
    )

    args = parser.parse_args()

    inn_data = Test_Dataset(args.watermarked_image, "png")
    inn_loader = DataLoader(inn_data, batch_size=args.batch_size, shuffle=False, drop_last=True)

    acc_history = []

    with torch.no_grad():
        if args.noise_type in ["JPEG", "HEAVY"]:
            fed_path = os.path.join("experiments", args.noise_type, "FED.pt")
            fed = FED().to(device)
            load(fed_path, fed)
            fed.eval()

            if args.noise_type == "HEAVY":
                inl_path = os.path.join("experiments", args.noise_type, "INL.pt")
                inl = INL().to(device)
                load(inl_path, inl)
                inl.eval()

            for idx, watermarked_images in enumerate(inn_loader):
                watermarked_images: Tensor = watermarked_images.to(device)
                embedded_messgaes: Tensor = torch.load(
                    os.path.join(args.messages_path, "message_{}.pt".format(idx + 1)),
                    map_location='cpu',
                    weights_only=True,
                ).to(device)

                all_zero = torch.zeros(embedded_messgaes.shape).to(device)

                if args.noise_type == "HEAVY":
                    watermarked_images = inl(watermarked_images.clone(), rev=True)

                reversed_img, extracted_messages = fed([watermarked_images, all_zero], rev=True)

                acc_rate = decoded_message_acc_rate(embedded_messgaes, extracted_messages)

                acc_history.append(acc_rate)

        else:
            raise ValueError("\"{}\" is not a valid noise type ".format(args.noise_type))

    print(f'acc: {np.mean(acc_history):.5f}%')


if __name__ == '__main__':
    main()
