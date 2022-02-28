import numpy as np
from radiomics import base, firstorder

# Computes Vx statistic
def compute_Vx(targetVoxelArray, dose):
    # Mask of voxels that receives at least 'dose' Gy 
    msk_above_dose = targetVoxelArray >= dose
    # Mask of valid voxels
    msk_valid_voxels = ~np.isnan(targetVoxelArray)

    return np.nansum(msk_above_dose, 1) / np.nansum(msk_valid_voxels, 1) * 100.

# Computes Dx statistic
def compute_Dx(targetVoxelArray, volume):
    # Quantile based on volume
    quantile = 100. - volume

    return np.nanpercentile(targetVoxelArray, quantile, axis=1)

# Callers for the RadiomicsDosesVolumes class
class GetVxFeatureValue(object):
    def __init__(self, radiomics_object, dose):
        self.radiomics_object = radiomics_object
        self.dose = dose
    def __call__(self):
        return compute_Vx(self.radiomics_object.targetVoxelArray, self.dose)

class GetDxFeatureValue(object):
    def __init__(self, radiomics_object, volume):
        self.radiomics_object = radiomics_object
        self.volume = volume
    def __call__(self):
        return compute_Dx(self.radiomics_object.targetVoxelArray, self.volume)

class RadiomicsDosesVolumes(firstorder.RadiomicsFirstOrder):
    """
    Doses-volumes indicators to characterize the dose distribution of a patient treated by radiotherapy.

    Vx: Vx % (volume) of the ROI has received at least x Gy
    Dx: x % (volume) of the ROI has received at least Dx Gy
    """
    def __init__(self, inputImage, inputMask, **kwargs):
        super(RadiomicsDosesVolumes, self).__init__(inputImage, inputMask, **kwargs)
        # Create getVxFeatureValue and getDxFeatureValue methods automatically
        x_volumes = [1., 2., 5., 10., 15., 20., 25., 30., 40.]
        x_doses = [10., 20., 30., 40., 50., 60., 70., 80., 90., 95., 99.]
        for x in x_volumes:
            setattr(self, f"getV{int(x)}FeatureValue", GetVxFeatureValue(self, x))
        for x in x_doses:
            setattr(self, f"getD{int(x)}FeatureValue", GetDxFeatureValue(self, x))

    @classmethod
    def getFeatureNames(cls):
        x_volumes = [1., 2., 5., 10., 15., 20., 25., 30., 40.]
        x_doses = [10., 20., 30., 40., 50., 60., 70., 80., 90., 95., 99.]
        names_Vx = ["V01", "V05"] + [f"V{int(x)}" for x in x_volumes]
        names_Dx = ["D01", "D05"] + [f"D{int(x)}" for x in x_doses]
        all_features = names_Vx + names_Dx
        return {feature: False for feature in all_features}

    ## Vx statistics whose name differ from value
    def getV01FeatureValue(self):
        return compute_Vx(self.targetVoxelArray, 0.1)

    def getV05FeatureValue(self):
        return compute_Vx(self.targetVoxelArray, 0.5)

    ## Dx statistics whose name differ from value
    def getD01FeatureValue(self):
        return compute_Dx(self.targetVoxelArray, 0.1)

    def getD05FeatureValue(self):
        return compute_Dx(self.targetVoxelArray, 0.5)

