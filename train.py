import torch.nn
import wandb

from models.encoder_decoder import FED
from utils.datasets import c, get_testloader, get_trainloader
from utils.jpeg import JpegSS, JpegTest
from utils.metric import *
from utils.utils import *

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


def load(model: torch.nn.Module, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)


trainloader = get_trainloader()
testloader = get_testloader()

fed = FED(c.diff, c.message_length)
fed = fed.to(device)

mse_loss = torch.nn.MSELoss()
optim = torch.optim.Adam(fed.parameters(), lr=c.lr)
scheduler = torch.optim.lr_scheduler.LinearLR(
    optim, start_factor=1, end_factor=1e-3, total_iters=c.epochs
)

if c.train_continue:
    load(fed, c.MODEL_PATH + c.suffix)

setup_logger('train', 'logging', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')

noise_layer = JpegSS(50)
test_noise_layer = JpegTest(50)

if c.WANDB:
    wandb.login()
    wandb.init(project='IIW', dir='logging', config=c.wandb_config)
    wandb.watch(fed, criterion='all', log_freq=10)

step = 0
for i_epoch in range(c.epochs):

    loss_history = []
    stego_loss_history = []
    message_loss_history = []

    #################
    #     train:    #
    #################

    fed.train()
    for idx_batch, cover_img in enumerate(trainloader):
        step += 1
        cover_img: Tensor = cover_img.to(device)

        message = torch.Tensor(
            np.random.choice([-0.5, 0.5], (cover_img.shape[0], c.message_length))
        ).to(device)
        input_data = [cover_img, message]

        #################
        #    forward:   #
        #################

        stego_img, left_noise = fed(input_data)
        stego_noise_img = noise_layer(stego_img.clone())

        #################
        #   backward:   #
        ################

        guass_noise = torch.zeros(left_noise.shape).to(device)
        output_data = [stego_noise_img, guass_noise]
        re_img, re_message = fed(output_data, rev=True)

        stego_loss: Tensor = mse_loss(stego_img, cover_img)
        message_loss: Tensor = mse_loss(re_message, message)

        total_loss = c.message_weight * message_loss + c.stego_weight * stego_loss
        total_loss.backward()

        optim.step()
        optim.zero_grad()

        loss_history.append(total_loss.item())
        stego_loss_history.append(stego_loss.item())
        message_loss_history.append(message_loss.item())

        if c.WANDB:
            wandb.log(
                dict(
                    total_loss=total_loss.item(),
                    stego_loss=stego_loss.item(),
                    message_loss=message_loss.item(),
                ),
                step,
            )

    scheduler.step()

    epoch_losses = np.mean(loss_history)
    stego_epoch_losses = np.mean(stego_loss_history)
    message_epoch_losses = np.mean(message_loss_history)

    logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
    logger_train.info(
        f"Train epoch {i_epoch}:   "
        f'Loss: {epoch_losses.item():.4f} | '
        f'Stego_Loss: {stego_epoch_losses.item():.4f} | '
        f'Message_Loss: {message_epoch_losses.item():.4f} | '
    )

    #################
    #     val:      #
    #################
    with torch.inference_mode():
        stego_psnr_history = []
        acc_history = []

        fed.eval()
        for test_cover_img in testloader:
            test_cover_img: Tensor = test_cover_img.to(device)

            test_message = torch.Tensor(
                np.random.choice([-0.5, 0.5], (test_cover_img.shape[0], c.message_length))
            ).to(device)

            test_input_data = [test_cover_img, test_message]

            #################
            #    forward:   #
            #################

            test_stego_img, test_left_noise = fed(test_input_data)

            if c.noise_flag:
                test_stego_noise_img = test_noise_layer(test_stego_img.clone())

            #################
            #   backward:   #
            #################

            test_z_guass_noise = torch.zeros(test_left_noise.shape).to(device)

            test_output_data = [test_stego_noise_img, test_z_guass_noise]

            test_re_img, test_re_message = fed(test_output_data, rev=True)

            psnr_temp_stego = psnr(test_cover_img, test_stego_img, 255)

            acc_rate = decoded_message_acc_rate(test_message, test_re_message)

            stego_psnr_history.append(psnr_temp_stego)
            acc_history.append(acc_rate)

        stego_psnr = np.mean(stego_psnr_history)
        acc = np.mean(acc_history)
        logger_train.info(f'TEST:   PSNR_STEGO: {stego_psnr:.5f}dB | ' f'Acc: {acc:.5f}% | ')

        if c.WANDB:
            wandb.log(
                dict(
                    psnr=stego_psnr.item(),
                    acc=acc.item(),
                    learning_rate=scheduler.get_last_lr()[0],
                ),
                step,
            )

    if i_epoch > 0 and (i_epoch % c.SAVE_freq) == 0:
        torch.save(
            {'opt': optim.state_dict(), 'net': fed.state_dict()},
            f'{c.MODEL_PATH}fed_{i_epoch:03}_{stego_psnr:.5f}dB_{acc:.5f}%.pt',
        )


torch.save(
    {'opt': optim.state_dict(), 'net': fed.state_dict()},
    f'{c.MODEL_PATH}fed_{i_epoch:03}_{stego_psnr:.5f}dB_{acc:.5f}%.pt',
)

if c.WANDB:
    wandb.finish()
