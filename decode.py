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

    acc_history1 = []
    acc_history2_1 = []
    acc_history2_2 = []

    with torch.no_grad():
        if args.noise_type in ["JPEG", "HEAVY"]:
            # fed_path = os.path.join("experiments", args.noise_type, "FED.pt")
            fed_path = os.path.join(
                'experiments/1125_11:22:33/JPEGfed_420_36.91206dB_99.99349%.pt'
            )
            fed = FED().to(device)
            load(fed_path, fed)
            fed.eval()

            if args.noise_type == "HEAVY":
                inl_path = os.path.join("experiments", args.noise_type, "INL.pt")
                inl = INL().to(device)
                load(inl_path, inl)
                inl.eval()

            for idx, (watermarked_images1, watermarked_images2) in enumerate(inn_loader):
                watermarked_images1: Tensor = watermarked_images1.to(device)
                watermarked_images2: Tensor = watermarked_images2.to(device)
                embedded_messgaes1: Tensor = torch.load(
                    os.path.join(args.messages_path, "message_{}_1.pt".format(idx + 1)),
                    map_location='cpu',
                    weights_only=True,
                ).to(device)
                key1: Tensor = torch.load(
                    os.path.join(args.messages_path, "key_{}_1.pt".format(idx + 1)),
                    map_location='cpu',
                    weights_only=True,
                ).to(device)
                embedded_messgaes2: Tensor = torch.load(
                    os.path.join(args.messages_path, "message_{}_2.pt".format(idx + 1)),
                    map_location='cpu',
                    weights_only=True,
                ).to(device)
                key2: Tensor = torch.load(
                    os.path.join(args.messages_path, "key_{}_2.pt".format(idx + 1)),
                    map_location='cpu',
                    weights_only=True,
                ).to(device)

                all_zero = torch.zeros(embedded_messgaes1.shape).to(device)

                if args.noise_type == "HEAVY":
                    watermarked_images1 = inl(watermarked_images1.clone(), rev=True)

                _, extracted_messages1, _ = fed([watermarked_images1, all_zero, key1], rev=True)
                _, extracted_messages2_1, _ = fed([watermarked_images2, all_zero, key1], rev=True)
                _, extracted_messages2_2, _ = fed([watermarked_images2, all_zero, key2], rev=True)

                acc_rate1 = decoded_message_acc_rate(embedded_messgaes1, extracted_messages1)
                acc_rate2_1 = decoded_message_acc_rate(embedded_messgaes1, extracted_messages2_1)
                acc_rate2_2 = decoded_message_acc_rate(embedded_messgaes2, extracted_messages2_2)

                acc_history1.append(acc_rate1)
                acc_history2_1.append(acc_rate2_1)
                acc_history2_2.append(acc_rate2_2)

        else:
            raise ValueError("\"{}\" is not a valid noise type ".format(args.noise_type))

    print(
        f'acc1: {np.mean(acc_history1):.5f}% | acc2_1: {np.mean(acc_history2_1):.5f}% | acc2_2: {np.mean(acc_history2_2):.5f}%'
    )


if __name__ == '__main__':
    main()
