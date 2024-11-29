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


def constrained_loss_fn(x: Tensor) -> Tensor:
    x = x * 0.5 + 0.5
    constrained_loss = torch.where(x > 1, x - 1, 0) + torch.where(x < 0, torch.abs(x), 0)
    return torch.sum(constrained_loss) / 2 / constrained_loss.shape[0]


trainloader = get_trainloader()
testloader = get_testloader()

fed = FED(c.diff, c.message_length)
fed = fed.to(device)

mse_loss = torch.nn.MSELoss()
optim = torch.optim.Adam(fed.parameters(), lr=c.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

os.makedirs(c.MODEL_PATH)
c.MODEL_PATH = os.path.join(c.MODEL_PATH, 'JPEG')

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
    constrained_loss_history = []

    #################
    #     train:    #
    #################

    fed.train()
    for idx_batch, img1 in enumerate(trainloader):
        step += 1
        img1: Tensor = img1.to(device)

        message1 = torch.Tensor(
            np.random.choice([-0.5, 0.5], (img1.shape[0], c.message_length))
        ).to(device)
        key1 = torch.randint(0, 2, message1.shape, dtype=torch.int8).to(device) * 2 - 1
        input_data1 = [img1, message1, key1]

        #################
        #    forward:   #
        #################

        stego_img1, left_noise1, _ = fed(input_data1)

        message2 = torch.Tensor(
            np.random.choice([-0.5, 0.5], (stego_img1.shape[0], c.message_length))
        ).to(device)
        key2 = torch.randint(0, 2, message2.shape, dtype=torch.int8).to(device) * 2 - 1
        input_data2 = [stego_img1, message2, key2]

        stego_img2, left_noise2, _ = fed(input_data2)

        #################
        #   backward:   #
        ################

        guass_noise1 = torch.zeros(left_noise1.shape).to(device)
        output_data1 = [stego_img1, guass_noise1, key1]
        _, re_message1, _ = fed(output_data1, rev=True)

        guass_noise2 = torch.zeros(left_noise2.shape).to(device)
        output_data2_1 = [stego_img2, guass_noise2, key1]
        _, re_message2_1, _ = fed(output_data2_1, rev=True)
        output_data2_2 = [stego_img2, guass_noise2, key2]
        _, re_message2_2, _ = fed(output_data2_2, rev=True)

        stego_loss1: Tensor = mse_loss(stego_img1, img1)
        message_loss1: Tensor = mse_loss(re_message1, message1)
        constrained_loss1: Tensor = constrained_loss_fn(stego_img1)

        stego_loss2: Tensor = mse_loss(stego_img2, stego_img1)
        message_loss2_1: Tensor = mse_loss(re_message2_1, message1)
        message_loss2_2: Tensor = mse_loss(re_message2_2, message2)
        constrained_loss2: Tensor = constrained_loss_fn(stego_img2)

        stego_loss = stego_loss1 + stego_loss2
        message_loss = message_loss1 + message_loss2_1 + message_loss2_2
        constrained_loss = constrained_loss1 + constrained_loss2
        total_loss = (
            c.stego_weight * stego_loss
            + c.message_weight * message_loss
            + c.constrained_weight * constrained_loss
        )
        total_loss.backward()

        optim.step()
        optim.zero_grad()

        loss_history.append(total_loss.item())
        stego_loss_history.append(stego_loss.item())
        message_loss_history.append(message_loss.item())
        constrained_loss_history.append(constrained_loss.item())

        if c.WANDB:
            wandb.log(
                dict(
                    total_loss=total_loss.item(),
                    stego_loss1=stego_loss1.item(),
                    stego_loss2=stego_loss2.item(),
                    message_loss1=message_loss1.item(),
                    message_loss2_1=message_loss2_1.item(),
                    message_loss2_2=message_loss2_2.item(),
                    constrained_loss1=constrained_loss1.item(),
                    constrained_loss2=constrained_loss2.item(),
                ),
                step,
            )

    scheduler.step()

    epoch_losses = np.mean(loss_history)
    stego_epoch_losses = np.mean(stego_loss_history)
    message_epoch_losses = np.mean(message_loss_history)
    constrained_epoch_losses = np.mean(constrained_loss_history)

    logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
    logger_train.info(
        f"Train epoch {i_epoch}:   "
        f'Loss: {epoch_losses.item():.4f} | '
        f'Stego_Loss: {stego_epoch_losses.item():.4f} | '
        f'Message_Loss: {message_epoch_losses.item():.4f} | '
        f'Constrained_Loss: {constrained_epoch_losses.item():.4f} | '
    )

    #################
    #     val:      #
    #################
    with torch.inference_mode():
        stego_psnr_history1 = []
        stego_psnr_history2 = []
        acc_history1 = []
        acc_history2_1 = []
        acc_history2_2 = []

        fed.eval()
        for img1 in testloader:
            img1: Tensor = img1.to(device)

            test_message1 = torch.Tensor(
                np.random.choice([-0.5, 0.5], (img1.shape[0], c.message_length))
            ).to(device)
            key1 = torch.randint(0, 2, test_message1.shape, dtype=torch.int8).to(device) * 2 - 1

            test_input_data1 = [img1, test_message1, key1]

            #################
            #    forward:   #
            #################

            test_stego_img1, test_left_noise1, _ = fed(test_input_data1)

            test_message2 = torch.Tensor(
                np.random.choice([-0.5, 0.5], (test_stego_img1.shape[0], c.message_length))
            ).to(device)
            key2 = torch.randint(0, 2, test_message2.shape, dtype=torch.int8).to(device) * 2 - 1

            test_input_data2 = [test_stego_img1, test_message2, key2]

            test_stego_img2, test_left_noise2, _ = fed(test_input_data2)

            #################
            #   backward:   #
            #################

            test_z_guass_noise1 = torch.zeros(test_left_noise1.shape).to(device)

            test_output_data1 = [test_stego_img1, test_z_guass_noise1, key1]

            _, test_re_message1, _ = fed(test_output_data1, rev=True)

            test_z_guass_noise2 = torch.zeros(test_left_noise2.shape).to(device)
            test_output_data2_1 = [test_stego_img2, test_z_guass_noise2, key1]
            _, test_re_message2_1, _ = fed(test_output_data2_1, rev=True)
            test_output_data2_2 = [test_stego_img2, test_z_guass_noise2, key2]
            _, test_re_message2_2, _ = fed(test_output_data2_2, rev=True)

            psnr_temp_stego1 = psnr(img1, test_stego_img1, 255)
            psnr_temp_stego2 = psnr(test_stego_img2, test_stego_img1, 255)

            acc_rate1 = decoded_message_acc_rate(test_message1, test_re_message1)
            acc_rate2_1 = decoded_message_acc_rate(test_message1, test_re_message2_1)
            acc_rate2_2 = decoded_message_acc_rate(test_message2, test_re_message2_2)

            stego_psnr_history1.append(psnr_temp_stego1)
            stego_psnr_history2.append(psnr_temp_stego2)
            acc_history1.append(acc_rate1)
            acc_history2_1.append(acc_rate2_1)
            acc_history2_2.append(acc_rate2_2)

        stego_psnr1 = np.mean(stego_psnr_history1)
        stego_psnr2 = np.mean(stego_psnr_history2)
        stego_psnr = np.mean([stego_psnr1, stego_psnr2])
        acc1 = np.mean(acc_history1)
        acc2_1 = np.mean(acc_history2_1)
        acc2_2 = np.mean(acc_history2_2)
        acc = np.mean([acc1, acc2_1, acc2_2])
        logger_train.info(f'TEST:   PSNR_STEGO: {stego_psnr:.5f}dB | Acc: {acc1:.5f}% | ')

        if c.WANDB:
            wandb.log(
                dict(
                    psnr1=stego_psnr1.item(),
                    psnr2=stego_psnr2.item(),
                    acc1=acc1.item(),
                    acc2_1=acc2_1.item(),
                    acc2_2=acc2_2.item(),
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
