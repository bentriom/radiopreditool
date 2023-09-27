
import os, sys, logging, argparse
sys.path.append("../workflow/scripts")
import learning_vae

# Pour batch size = 32, zoom = x10 avec CNN VAE N=64; pas de out of memory de cuda
# Pour batch size = 32, zoom = x8 avec CNN VAE N=64; pas de out of memory de cuda
# Pour batch size = 32, zoom = x5 avec CNN VAE N=64; out of memory de cuda

if __name__ == "__main__":
    device = "cuda"
    RESULTS_DIR = "/workdir/bentrioum/results/radiopreditool/radiomics/"
    METADATA_DIR = f"{RESULTS_DIR}metadata/"
    NII_DL_DIR = RESULTS_DIR + f"nii/deep_learning/"
    FCCSS_CLINICAL_DATASET = "/workdir/bentrioum/database/radiopreditool/fccss/base_fccss_igr_curie_011222_extended.csv.gz"
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch-size", nargs='?', help="Batch size during training", default=16, type=int)
    arg_parser.add_argument("--vae", nargs='?', help="The VAE architecture", default="N64_2", type=str)
    arg_parser.add_argument("--zoom", nargs='?', help="The zoom scale during training", default=10, type=int)
    arg_parser.add_argument("--start-epoch", nargs='?', help="The starting epoch", default=0, type=int)
    arg_parser.add_argument("--nepochs", nargs='?', help="The number of epochs", default=10, type=int)
    args = arg_parser.parse_args()
    cvae_type, image_zoom = args.vae, args.zoom
    batch_size, n_epochs, start_epoch = args.batch_size, args.nepochs,  args.start_epoch
    print(batch_size)
    print(start_epoch)
    print(args)
    VAE_CONFIG = {"CVAE_TYPE": cvae_type, "IMAGE_ZOOM": image_zoom, "BATCH_SIZE": batch_size,
    "N_EPOCHS": n_epochs, "START_EPOCH": start_epoch}
    VAE_DIR = RESULTS_DIR + f"VAE/{VAE_CONFIG['CVAE_TYPE']}/zoom_x{VAE_CONFIG['IMAGE_ZOOM']}/"
    learning_vae.run_learn_vae(METADATA_DIR, VAE_DIR, file_fccss_clinical = FCCSS_CLINICAL_DATASET,
                               cvae_type = VAE_CONFIG["CVAE_TYPE"],
                               downscale = VAE_CONFIG["IMAGE_ZOOM"], batch_size = VAE_CONFIG["BATCH_SIZE"],
                               n_epochs = VAE_CONFIG["N_EPOCHS"], start_epoch = VAE_CONFIG["START_EPOCH"],
                               device = device, log_level = logging.DEBUG, log_name = "learn_vae")

