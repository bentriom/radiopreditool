
import pandas as pd
import numpy as np
import math, os, random
import nibabel
from scipy import ndimage
from torch.utils.data import Dataset

# A class that creates a PyTorch dataset for nifti images from FCCSS newdosi dataset
class FccssNewdosiDataset(Dataset):
    def __init__(self, metadata_dir, file_fccss_clinical = None, max_scale = 85.306, with_index = False,
                       train_size = 0.7, phase = "data", downscale = 1, seed_sample = 21):
        assert max_scale != 0
        assert 0 < train_size < 1
        assert os.path.isdir(metadata_dir)
        if file_fccss_clinical is not None:
            assert os.path.isfile(file_fccss_clinical)
        self.phase = phase
        self.downscale = downscale
        self.max_scale = max_scale
        self.with_index = with_index
        # Get input size of images with meta data
        df_size = pd.read_csv(metadata_dir + "biggest_image_size.csv", index_col = 0, names = ["size"], header = None)
        biggest_image_size = df_size.loc[["size_x", "size_y", "size_z"], "size"].values
        self.input_image_size = ndimage.zoom(np.zeros(biggest_image_size[[2,1,0]]), 1/downscale, order=0).shape
        # Get the list of images to load when __getitem__ is called
        df_images_paths = pd.read_csv(metadata_dir + "images_paths_dl.csv")
        df_images_paths = df_images_paths.loc[df_images_paths["size_bytes"] > 0, :]
        if file_fccss_clinical is not None:
            df_fccss = pd.read_csv(file_fccss_clinical, low_memory = False)[["ctr", "numcent"]]
            df_images_paths = df_images_paths.merge(df_fccss, how = "inner", on = ["ctr", "numcent"])
        self.df_images_paths = df_images_paths
        # Generate train/test indexes
        nbr_samples_train = int(train_size * len(df_images_paths.index))
        random.seed(seed_sample)
        train_idx = random.sample(sorted(df_images_paths.index), k = nbr_samples_train)
        if phase == "data":
            self.images_paths = df_images_paths.loc[:, "absolute_path"].values
            self.list_ctr = df_images_paths.loc[:, "ctr"].astype(int).values
            self.list_numcent = df_images_paths.loc[:, "numcent"].astype(int).values
        elif phase == "train":
            self.images_paths = df_images_paths.loc[train_idx, "absolute_path"].values
            self.list_ctr = df_images_paths.loc[train_idx, "ctr"].astype(int).values
            self.list_numcent = df_images_paths.loc[train_idx, "numcent"].astype(int).values
        elif phase == "test":
            test_idx_mask = ~df_images_paths.index.isin(train_idx)
            self.images_paths = df_images_paths.loc[test_idx_mask, "absolute_path"].values
            self.list_ctr = df_images_paths.loc[test_idx_mask, "ctr"].astype(int).values
            self.list_numcent = df_images_paths.loc[test_idx_mask, "numcent"].astype(int).values
        elif phase == "extraction":
+            self.images_paths = df_images_paths.loc[:, "absolute_path"].values
+            self.list_ctr = df_images_paths.loc[:, "ctr"].astype(int).values
+            self.list_numcent = df_images_paths.loc[:, "numcent"].astype(int).values
        else:
            raise NameError("Phase in FccssNewdosiDataset is not recognized.")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        # read image
        image_path = self.images_paths[idx]
        image = nibabel.load(image_path)
        assert image is not None
        # data processing
        image_array = self.__data_process__(image)
        # convert as tensor array
        image_tensor = self.__nii2tensorarray__(image_array)
        if self.phase in ["data", "train", "test"]:
            if self.with_index:
                return image_tensor, idx
            else:
                return image_tensor
        elif self.phase == "extraction":
            ctr, numcent = self.list_ctr[idx], self.list_numcent[idx]
            return image_tensor, ctr, numcent

    def __data_process__(self, data):
        new_image_array = data.get_fdata()
        if self.max_scale is not None:
            new_image_array = self.__minmax_scale__(new_image_array, -1, self.max_scale)
        new_image_array = self.__resize_data__(new_image_array)
        return new_image_array

    def __minmax_scale__(self, image_array, min_scale, max_scale, min_range = -1, max_range = 1):
        scaled_image = (image_array - min_scale) / (max_scale - min_scale)
        return scaled_image * (max_range - min_range) + min_range

    def __resize_data__(self, image_array):
        if self.downscale == 1:
            return image_array
        else:
            scaled_image_array = ndimage.zoom(image_array, 1/self.downscale, order=0)
            return scaled_image_array

    def __nii2tensorarray__(self, data):
        [dim_x, dim_y, dim_z] = data.shape
        new_data = np.reshape(data, [1, dim_x, dim_y, dim_z])
        new_data = new_data.astype("float32")

        return new_data

    def __eliminate_outliers__(self, image_array):
        D98 = np.nanpercentile(image_array, 2)
        D2 = np.nanpercentile(image_array, 98)
        image_array[image_array > D2] = D2
        image_array[image_array < D98] = D98

        return image_array

    def get_image_array(self, idx):
        # read image
        image_path = self.images_paths[idx]
        image_label = 0
        image = nibabel.load(image_path)
        assert image is not None
        # data processing
        image_array = self.__data_process__(image)

        return image_array

    def save_processed_nii(self, idx, save_dir):
        os.makedirs(save_dir, exist_ok = True)
        image_array = self.get_image_array(idx)
        image_name = os.path.basename(self.images_paths[idx]).replace(".nii.gz", f"_downscale_{self.downscale}.nii.gz")
        image_nii = nibabel.Nifti1Image(image_array, np.eye(4))
        nibabel.save(image_nii, save_dir + image_name)

