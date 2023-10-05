
import os, sys, logging
import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

import pytorch_dataset as pdata
from vae import *
from radiopreditool_utils import *
from viz import plot_loss_vae

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

def vae_loss(x_hat, x, mu, logvar, mse_scale, kl_weight):
    MSE = torch.nn.MSELoss(reduction='sum')(x_hat, x).mul(mse_scale).div(float(len(x)))
    # MSE = torch.nn.MSELoss(reduction='mean')(x_hat, x)
    KLD = torch.tensor(0.0, requires_grad = True)
    if kl_weight > 0:
        KLD =  torch.sum(1 + logvar - mu.pow(2) - logvar.exp()).mul(-0.5*kl_weight).div(float(len(x)))

    return MSE + KLD, MSE, KLD

def train_loop(epoch, model, train_dataloader, mse_scale, kl_weight, optimizer, device, scheduler, log_name):
    model.train()
    train_total_loss = 0
    train_MSE_loss = 0
    train_KLD_loss = 0
    logger = logging.getLogger(log_name)
    if logger.level <= logging.DEBUG:
        norm_grad = 0
        params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        for p in params:
            norm_p_grad = p.grad.detach().data.norm(2)
            norm_grad += norm_p_grad.item() ** 2
        norm_grad = norm_grad ** 0.5
        logger.debug(f"Gradient norm ({len(params)} params): {norm_grad}")
    # for batch_idx, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='train'):
    for batch_idx, data in enumerate(train_dataloader):
        # Load data according to the dataset mode (if it gives also the index or not)
        indexes = None
        if train_dataloader.dataset.with_index:
            indexes = data[1]
            data = data[0]
        logger.info(f"# Batch train {batch_idx}/{len(train_dataloader)-1} of size {len(data)}")
        logger.debug(f"-- indexes of the batch: {indexes}")
        if logger.level <= logging.DEBUG:
            if torch.any(torch.isnan(data)):
                logger.debug("Batch contains NaN")
                mask_nan_images = [i for i, image in enumerate(data) if torch.any(torch.isnan(image))]
                logger.debug(f"NaN img indexes: {np.asarray(indexes)[mask_nan_images]}")
        flush_log(logger)
        # compute model output
        logger.debug(f"-- estimated size in GB: {(data.element_size() * data.numel())/10**9}")
        data = data.to(device, dtype=torch.float)
        logger.debug(f"-- loaded on device {device}")
        optimizer.zero_grad()
        batch_x_hats, mu, logvar, _ = model(data)
        # if logger.level <= logging.DEBUG:
        if logger.level <= logging.DEBUG:
            if torch.any(torch.isnan(batch_x_hats)):
                logger.debug("-- /!\ reconstruction images have NaN")
                mask_nan_batch = [torch.any(torch.isnan(image_hat)) for image_hat in batch_x_hats]
                if indexes is not None:
                    logger.debug(f"-- images whose reconstruction is NaN in the batch:"
                                 f"{np.asarray(indexes)[mask_nan_batch]}")
                dict_results = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                checkpoint_file = f"./last_state_before_crash.pth"
                torch.save(dict_results, checkpoint_file)
        logger.debug(f"-- output computed by the model")
        # compute batch losses
        total_loss, MSE_loss, KLD_loss = vae_loss(batch_x_hats, data, mu, logvar, mse_scale, kl_weight)
        train_total_loss += total_loss.item()
        train_MSE_loss += MSE_loss.item()
        train_KLD_loss += KLD_loss.item()
        logger.debug(f"-- losses computed: MSE={MSE_loss} KLD={KLD_loss} total={total_loss}")
        assert not np.isnan(total_loss.item()), "Loss is NaN"
        # compute gradients and update weights
        total_loss.backward()
        if logger.level <= logging.DEBUG:
            norm_grad = 0
            params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
            for p in params:
                p_norm = p.grad.detach().data.norm(2)
                norm_grad += p_norm.item() ** 2
            norm_grad = norm_grad ** 0.5
            logger.debug(f"-- gradient computed ({len(params)} params), norm: {norm_grad}")
        optimizer.step()
        # schedule learning rate
        scheduler.step()
        logger.debug(f"-- learning rate computed: {optimizer.param_groups[0]['lr']}")

    train_total_loss /= len(train_dataloader)
    train_MSE_loss /= len(train_dataloader)
    train_KLD_loss /= len(train_dataloader)

    return train_total_loss, train_MSE_loss, train_KLD_loss

def test_loop(epoch, model, test_dataloader, mse_scale, kl_weight, device, log_name):
    model.eval()
    test_total_loss = 0
    test_MSE_loss = 0
    test_KLD_loss = 0
    logger = logging.getLogger(log_name)
    mse_scale = 1
    with torch.no_grad():
        # for batch_idx, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='test'):
        for batch_idx, data in enumerate(test_dataloader):
            # Load data according to the dataset mode (if it gives also the index or not)
            indexes = None
            if test_dataloader.dataset.with_index:
                indexes = data[1]
                data = data[0]
            # compute loss
            logger.info(f"# Batch test {batch_idx}/{len(test_dataloader)-1}")
            flush_log(logger)
            data = data.to(device, dtype=torch.float)
            logger.debug(f"-- loaded on device {device}")
            batch_x_hats, mu, logvar, latent_batch = model(data)
            logger.debug(f"-- output computed by the model")
            total_loss, MSE_loss, KLD_loss = vae_loss(batch_x_hats, data, mu, logvar, mse_scale, kl_weight)
            logger.debug(f"-- losses computed: MSE={MSE_loss} KLD={KLD_loss} total={total_loss}")

            test_total_loss += total_loss.item()
            test_MSE_loss += MSE_loss.item()
            test_KLD_loss += KLD_loss.item()

    test_total_loss /= len(test_dataloader)
    test_MSE_loss /= len(test_dataloader)
    test_KLD_loss /= len(test_dataloader)

    return test_total_loss, test_MSE_loss, test_KLD_loss

def setup_gpu(rank_process, number_of_processes):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank = rank_process, world_size = number_of_processes)

def cleanup_gpu():
    dist.destroy_process_group()

def learn_vae(rank_device, nb_devices, metadata_dir, vae_dir, file_fccss_clinical, cvae_type, downscale,
              batch_size, n_epochs, start_epoch, log_name, log_level):
    assert isinstance(rank_device, int) or rank_device in ["mps", "cpu"]
    is_cuda = rank_device not in ["mps", "cpu"]
    # Setup distributed module for cuda device
    if is_cuda:
        setup_gpu(rank_device, nb_devices)
    save_epochs_dir = f"{vae_dir}epochs/"
    os.makedirs(vae_dir, exist_ok = True)
    os.makedirs(save_epochs_dir, exist_ok = True)
    log_name_device = f"{log_name}_{rank_device}"
    if log_name is not None:
        logger = setup_logger(log_name_device, vae_dir + f"{log_name}_device_{rank_device}.log", level = log_level)
    else:
        logger = setup_logger(log_name_device, None, level = log_level)
    # If we run on cpu, we only need to log in the main file
    if rank_device == "cpu":
        logger = logging.getLogger(log_name)
        log_name_device = log_name
    logger.info(f"Learning convolutional VAE {cvae_type} on the device {rank_device}.")
    logger.info(f"Image zoom: {downscale}, batch size = {batch_size}")
    flush_log(logger)
    # Datasets
    trainset = pdata.FccssNewdosiDataset(metadata_dir, file_fccss_clinical = file_fccss_clinical, with_index = True,
                                         phase = "train", downscale = downscale)
    testset = pdata.FccssNewdosiDataset(metadata_dir, file_fccss_clinical = file_fccss_clinical, with_index = True,
                                        phase = "test", downscale = downscale)
    train_dataloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, pin_memory = True)
    test_dataloader = DataLoader(testset, shuffle = True, pin_memory = True)
    logger.info(f"Dataset loader created. Input image size: {trainset.input_image_size}.")
    # Get CNN model
    if cvae_type == "N128":
        cnn_vae = CVAE_3D_N128(image_channels = 1, z_dim = 16, input_image_size = trainset.input_image_size)
    if cvae_type == "N64":
        cnn_vae = CVAE_3D_N64(image_channels = 1, z_dim = 8, input_image_size = trainset.input_image_size)
    if cvae_type == "N64_2":
        cnn_vae = CVAE_3D_N64_2(image_channels = 1, kernel_size = 3, leakyrelu_slope = 0.2,
                                z_dim = 8, input_image_size = trainset.input_image_size)
    if cvae_type == "N32_2":
        cnn_vae = CVAE_3D_N32_2(image_channels = 1, kernel_size = 3, leakyrelu_slope = 0.2,
                                z_dim = 8, input_image_size = trainset.input_image_size)
    logger.info(f"CNN VAE {cvae_type} loaded.")
    logger.info(f"Latent dim: {cnn_vae.z_dim}. Kernel size: {cnn_vae.kernel_size}."
                f"LeakyReLU slope: {cnn_vae.leakyrelu_slope}")
    cnn_vae.init_weights()
    logger.info(f"Model weights are initialized.")
    cnn_vae.to(rank_device)
    if is_cuda:
        cnn_vae = DistributedDataParallel(cnn_vae, device_ids = [rank_device])
    logger.info(f"Model loaded on device {rank_device}.")
    flush_log(logger)
    # print("Computing a forward")
    # train_batch0 = next(iter(train_dataloader))
    # cnn_vae.forward(train_batch0[0])

    # Set optimizer and best test loss based on starting epoch
    optimizer = optim.Adam(cnn_vae.parameters(), lr=1e-4)
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
    kl_weights = schedule_KL_annealing(0.0, 1.0, n_epochs, 5)
    logger.info(f"Scheduled KLD weights: {kl_weights}")
    # We scale the MSE with sum reduction to a cube of shape 16x16x16
    mse_scale = (16 ** 3) / np.asarray(trainset.input_image_size).prod()
    logger.info(f"Reconstruction error (MSE) scale: {mse_scale}")

    # Training loop
    logger.info("Starting epoch training")
    for epoch in range(start_epoch, n_epochs):
        logger.info("Epoch {}".format(epoch))
        # Update KL weight at every epoch
        kl_weight = kl_weights[epoch]
        logger.info(f"Current KL weight: {kl_weight}")
        # Train losses
        train_total_loss, train_MSE_loss, train_KLD_loss = train_loop(epoch, cnn_vae, train_dataloader, mse_scale,
                                                                      kl_weight, optimizer, rank_device,
                                                                      scheduler, log_name_device)
        logger.info("Epoch [%d/%d] train_total_loss: %.3f, train_REC_loss: %.3f, train_KLD_loss: %.3f" \
                    % (epoch, n_epochs, train_total_loss, train_MSE_loss, train_KLD_loss))
        flush_log(logger)
        # Test losses
        test_total_loss, test_MSE_loss, test_KLD_loss = test_loop(epoch, cnn_vae, test_dataloader, mse_scale,
                                                                  kl_weight, rank_device, log_name_device)
        logger.info("Epoch [%d/%d] test_total_loss: %.3f, test_REC_loss: %.3f, test_KLD_loss: %.3f" \
                    % (epoch, n_epochs, test_total_loss, test_MSE_loss, test_KLD_loss))
        flush_log(logger)

        best_test_loss = min(test_total_loss, best_test_loss)
        dict_results_epoch = {
            'epoch': epoch,
            'train_total_loss': train_total_loss,
            'train_MSE_loss': train_MSE_loss,
            'train_KLD_loss': train_KLD_loss,
            'test_total_loss': test_total_loss,
            'test_MSE_loss': test_MSE_loss,
            'test_KLD_loss': test_KLD_loss,
            'best_test_loss': best_test_loss,
            'state_dict': cnn_vae.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if is_cuda:
            if rank_device == 0:
                save_epoch(dict_results_epoch, save_dir = save_epochs_dir)
        else:
            save_epoch(dict_results_epoch, save_dir = save_epochs_dir)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataloader))
    # Cleanup distributed module for cuda device
    if is_cuda:
        cleanup_gpu(rank_device, nb_devices)

def run_learn_vae(metadata_dir, vae_dir, file_fccss_clinical = None, cvae_type = "N128", downscale = 1,
                  batch_size = 64, n_epochs = 10, start_epoch = 0, device = "cpu",
                  log_level = logging.INFO, log_name = "learn_vae"):
    assert device in ["cpu", "mps", "cuda"]
    assert cvae_type in ["N32_2", "N64", "N64_2", "N128"]
    assert 0 <= start_epoch < n_epochs
    os.makedirs(vae_dir, exist_ok = True)
    logger = setup_logger(log_name, vae_dir + f"{log_name}.log", level = log_level)
    # Checks if the chosen device is available on the machine
    if device == "cuda" and not torch.backends.cuda.is_built():
        raise ValueError("Torch device is set on cuda but it is not build on this machine.")
    if device == "mps" and not torch.backends.mps.is_built():
        raise ValueError("Torch sevice is set on mps but it is not build on this machine.")
    # If CUDA, Spawning the learning processes over possibly multiple gpus
    if device == "cuda":
        logger.info(f"Number of devices visible by cuda: {torch.cuda.device_count()}")
        if is_slurm_run():
            assert "CUDA_VISIBLE_DEVICES" in os.environ
            logger.info(f"Cuda visible devices on slurm: {os.environ['CUDA_VISIBLE_DEVICES']}")
            device_ids = [int(device_id) for device_id in os.environ["CUDA_VISIBLE_DEVICES"].split(',')]
        else:
            device_ids = range(torch.cuda.device_count())
        mp.spawn(learn_vae, args = (len(device_ids), metadata_dir, vae_dir, file_fccss_clinical, cvae_type,
                                    downscale, batch_size, n_epochs, start_epoch, log_name, log_level), nprocs = len(device_ids))
    else:
        learn_vae(device, 1, metadata_dir, vae_dir, file_fccss_clinical, cvae_type, downscale,
                  batch_size, n_epochs, start_epoch, log_name, log_level)
    logger.info(f"Learning completed")
    plot_loss_vae(vae_dir)

