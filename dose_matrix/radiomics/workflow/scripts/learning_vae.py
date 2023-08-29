
import os, sys, logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_dataset as pdata
from vae import *
from radiopreditool_utils import *

def save_epoch(nn_state_epoch, save_dir = "./"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint_file = os.path.join(save_dir, f"epoch_{nn_state_epoch['epoch']}.pth")
    torch.save(nn_state_epoch, checkpoint_file)

def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
    """
    Custom function for multiple annealing scheduling: Monotonic and cyclical_annealing
    Given number of epochs, it returns the value of the KL weight at each epoch as a list.
    Based on: https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    """
    weights = np.ones(n_epochs)
    period = n_epochs/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epochs):
            weights[int(i+c*period)] = v
            v += step
            i += 1

    return weights

def vae_loss(x_hat, x, mu, logvar, kl_weight):
    MSE = 0.1 * torch.nn.MSELoss(reduction='sum')(x_hat, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return MSE + KLD, MSE, KLD

def train_loop(epoch, model, train_dataloader, kl_weight, optimizer, device, scheduler, log_name = "learn_vae"):
    model.train()
    train_total_loss = 0
    train_BCE_loss = 0
    train_KLD_loss = 0
    logger = logging.getLogger(log_name)
    # for batch_idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='train'):
    for batch_idx, data in enumerate(train_dataloader):
        logger.info(f"Batch train {batch_idx}/len(train_dataloader)")
        # compute model output
        data = data.to(device, dtype=torch.float)
        optimizer.zero_grad()
        batch_x_hats, mu, logvar, _ = model(data)
        # compute batch losses
        total_loss, BCE_loss, KLD_loss = vae_loss(batch_x_hats, data, mu, logvar, kl_weight)
        train_total_loss += total_loss.item()
        train_BCE_loss += BCE_loss.item()
        train_KLD_loss += KLD_loss.item()
        # compute gradients and update weights
        total_loss.backward()
        optimizer.step()
        # schedule learning rate
        scheduler.step()

    train_total_loss /= len(train_dataloader.dataset)
    train_BCE_loss /= len(train_dataloader.dataset)
    train_KLD_loss /= len(train_dataloader.dataset)

    return train_total_loss, train_BCE_loss, train_KLD_loss

def test_loop(epoch, model, test_dataloader, kl_weight, device, log_name = "learn_vae"):
    model.eval()
    test_total_loss = 0
    test_BCE_loss = 0
    test_KLD_loss = 0
    logger = logging.getLogger(log_name)

    with torch.no_grad():
        # for batch_idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='test'):
        for batch_idx, data in enumerate(test_dataloader):
            # compute loss
            logger.info(f"Batch test {batch_idx}/len(test_dataloader)")
            data = data.to(device, dtype=torch.float)
            batch_x_hats, mu, logvar, latent_batch = model(data)
            total_loss, BCE_loss, KLD_loss = vae_loss(batch_x_hats, data, mu, logvar, kl_weight)

            test_total_loss += total_loss.item()
            test_BCE_loss += BCE_loss.item()
            test_KLD_loss += KLD_loss.item()

    test_total_loss /= len(test_dataloader.dataset)
    test_BCE_loss /= len(test_dataloader.dataset)
    test_KLD_loss /= len(test_dataloader.dataset)

    return test_total_loss, test_BCE_loss, test_KLD_loss

def learn_vae(metadata_dir, vae_dir, n_channels_end = 128, downscale = 1, batch_size = 64,
              n_epochs = 10, test_every_epochs = 1, start_epoch = 0, device = "cpu", log_stdout = False):
    assert device in ["cpu", "mps", "cuda"]
    assert n_channels_end in [64, 128]
    assert 0 <= start_epoch < n_epochs
    save_epochs_dir = f"{vae_dir}epochs/"
    os.makedirs(vae_dir, exist_ok = True)
    os.makedirs(save_epochs_dir, exist_ok = True)
    log_name = "learn_vae"
    logger = setup_logger(log_name, vae_dir + f"{log_name}.log")
    logger.info(f"Learning convolutional VAE N={n_channels_end}")
    logger.info(f"Image zoom: {downscale}, batch size = {batch_size}")
    # Datasets
    trainset = pdata.FccssNewdosiDataset(metadata_dir, phase = "train", downscale = downscale)
    testset = pdata.FccssNewdosiDataset(metadata_dir, phase = "test", downscale = downscale)
    train_dataloader = DataLoader(trainset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(testset, shuffle = False)
    logger.info(f"Dataset loader created. Input image size: {trainset.input_image_size}.")
    # CNN model
    if n_channels_end == 128:
        cnn_vae = CVAE_3D_N128(image_channels = 1, z_dim = 32, input_image_size = trainset.input_image_size)
    if n_channels_end == 64:
        cnn_vae = CVAE_3D_N64(image_channels = 1, z_dim = 32, input_image_size = trainset.input_image_size)
    if device == "mps" and not torch.backends.mps.is_built():
        raise ValueError("Torch sevice is set on mps but it is not build on this machine.")
    if device == "cuda" and not torch.backends.cuda.is_built():
        raise ValueError("Torch device is set on cuda but it is not build on this machine.")
    logger.info(f"CNN VAE created.")
    cnn_vae.to(device)
    logger.info("Model loaded on {device}.")
    if device == "cuda":
        logger.info(f"Number of devices visible by cuda: {torch.cuda.device_count()}")
    # print("Computing a forward")
    # train_batch0 = next(iter(train_dataloader))
    # cnn_vae.forward(train_batch0[0])

    # Set optimizer and best test loss based on starting epoch
    optimizer = optim.Adam(cnn_vae.parameters(), lr=1e-3) # 1e-4 0 KLD, 1e-3 works, 1e-1 & 1e-2 gives NaN
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))
    best_test_loss = np.finfo('f').max
    if start_epoch > 0:
        n_epoch_load = start_epoch - 1
        logger.info(f"Loading training results from epoch {n_epoch_load}")
        nn_state = torch.load(f"{save_epochs_dir}epoch_{n_epoch_load}.pth")
        assert nn_state["epoch"] == n_epoch_load
        cnn_vae.load_state_dict(nn_state['state_dict'])
        best_test_loss = nn_state['best_test_loss']
        optimizer.load_state_dict(nn_state['optimizer'])
    # Schedule KL annealing
    kl_weights = schedule_KL_annealing(0.0, 1.0, n_epochs, 4)
    kl_weight = 0

    # Training loop
    logger.info("Starting epoch training")
    for epoch in range(start_epoch, n_epochs):
        logger.info("Epoch {}".format(epoch))
        # Update KL weight at every epoch
        kl_weight = kl_weights[epoch]
        logger.info(f"Current KL weight: {kl_weight}")
        # Train losses
        train_total_loss, train_BCE_loss, train_KLD_loss = train_loop(epoch, cnn_vae, train_dataloader,
                                                                      kl_weight, optimizer, device, scheduler)
        logger.info("Epoch [%d/%d] train_total_loss: %.3f, train_REC_loss: %.3f, train_KLD_loss: %.3f" \
                    % (epoch, n_epochs, train_total_loss, train_BCE_loss, train_KLD_loss))
        # Test losses
        if (epoch % test_every_epochs == 0) or (epoch == n_epochs-1):
            test_total_loss, test_BCE_loss, test_KLD_loss = test_loop(epoch, cnn_vae, test_dataloader, kl_weight, device)
            logger.info("Epoch [%d/%d] test_total_loss: %.3f, test_REC_loss: %.3f, test_KLD_loss: %.3f" \
                        % (epoch, n_epochs, test_total_loss, test_BCE_loss, test_KLD_loss))

            best_test_loss = min(test_total_loss, best_test_loss)
            save_epoch({
                'epoch': epoch,
                'best_test_loss': best_test_loss,
                'state_dict': cnn_vae.state_dict(),
                'optimizer': optimizer.state_dict(),
                }, save_dir = save_epochs_dir)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))

