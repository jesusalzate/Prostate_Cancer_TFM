import pandas as pd
import numpy as np
import os
import json
from numpy import std,mean
from radiomics import featureextractor
import SimpleITK as sitk
from multiprocessing import cpu_count, Pool
import radiomics 
import logging

# Set up logging
logging.basicConfig(filename="Picai_train_radiomics_log_t2w.txt",
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up radiomics logger for detailed logging
rLogger = logging.getLogger("radiomics")
rLogger.setLevel(logging.INFO)


def delete_black_slices(image: sitk.Image, threshold: float = 0.5):
    #image = sitk.ReadImage(image_path)
    
    data = sitk.GetArrayFromImage(image)
    img_std = std(data, keepdims=False)
    std_per_slice_axis1 = std(data, axis=(1, 2), keepdims=False)
    std_per_slice_axis2 = std(data, axis=(0, 2), keepdims=False)
    std_per_slice_axis3 = std(data, axis=(0, 1), keepdims=False)

    img_mean = mean(data, keepdims=False)
    mean_per_slice_axis1 = mean(data, axis=(1, 2), keepdims=False)
    mean_per_slice_axis2 = mean(data, axis=(0, 2), keepdims=False)
    mean_per_slice_axis3 = mean(data, axis=(0, 1), keepdims=False)

    mask_axis1 = (std_per_slice_axis1 > threshold * img_std) & (mean_per_slice_axis1 > threshold * img_mean)
    mask_axis2 = (std_per_slice_axis2 > threshold * img_std) & (mean_per_slice_axis2 > threshold * img_mean)
    mask_axis3 = (std_per_slice_axis3 > threshold * img_std) & (mean_per_slice_axis3 > threshold * img_mean)

    mask = np.full(data.shape, False)

    mask[np.ix_(mask_axis1, mask_axis2, mask_axis3)] = True
    # mask[mask_axis1, mask, :] = True
    # mask[:,mask_axis2,:] = True
    # mask[:,:,mask_axis3] = True
    masked_image = data[mask_axis1, :, :]
    masked_image = data[:, mask_axis2, :]
    masked_image = data[:, :, mask_axis3]
    
    
    #convert mask to simpleitk image
    mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    mask.CopyInformation(image)

    #sitk.WriteImage(mask, "Prostate_Cancer_TFM/Data_Analysis/EHRtemporalVariability/tmp_mask_radiomics/tmp_mask.nii.gz")
    return mask, masked_image


def calculate_radiomics(case): 
    rLogger.info("Processing Patient: %s", case["ID"])
    if os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings["binWidth"] = 25
        settings["resampledPixelSpacing"] = [0.5,0.5,3]
        settings["interpolator"] = sitk.sitkBSpline
        settings["correctMask"] = True
        settings["normalize"] = True
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    print(
        f"Processing Patient {case['ID']} (Image: {case['Image']})"
    )

    image_path = case["Image"]
    image = sitk.ReadImage(image_path)

    mask,masked_image = delete_black_slices(image, threshold=0.5)
    
    values = np.unique(sitk.GetArrayFromImage(mask))
    #print(case['ID'],mask.GetSize())
    #print(case['ID'],values)
    #if len of values is 1, then zero padd the mask and the image
    if len(values) == 1:
        try:
            mask = sitk.ConstantPad(mask, (1,1,1), (1,1,1))
            image = sitk.ConstantPad(image, (1,1,1), (1,1,1))
            values = np.unique(sitk.GetArrayFromImage(mask))
            #print(case['ID'],values)
            #print(case['ID'],mask.GetSize())
        except Exception as e:
            print("Error zero padding the mask and image", str(e))
    
    #print(values)
    # Uncomment if you want to check the number and values of all labels in the mask
    # print(values)
    values = np.array([1])  # Specify the values of labels of interest
    # Uncomment if you want to analyse all labels of mask
    # values = np.delete(values, 0)
    patient = []
    for index, label in enumerate(values, start=1):
        label = int(label)
        rLogger.info(
            "Processing Patient %s (Image: %s, Label: %s)",
            case["ID"],
            case["Image"],
            label,
        )
        if (image_path is not None):
            try:
                result = pd.Series(extractor.execute(image, mask, label))
            except Exception:
                rLogger.error("FEATURE EXTRACTION FAILED:", exc_info=True)
                rLogger.error("Skipping Patient %s", case["ID"])
                result = pd.Series()
        else:
            rLogger.error("FEATURE EXTRACTION FAILED: Missing Image and/or Mask")
            result = pd.Series()

        result.name = case["ID"]
        result = result.add_prefix("label{}_".format(index))
        patient.append(result)
    if len(patient) == 0:
        rLogger.error(f"FEATURE EXTRACTION FAILED: {case['ID']}")
        patient = pd.Series()
        patient.name = case["ID"]
    elif len(patient) == 1:
        patient = patient[0]
    else:
        patient = pd.concat(patient, axis=0)

    return patient

# Path to the data

path = "/mnt/ceib/datalake/FISABIO_datalake/prueba/p0052021_reborn"
derivatives_path = "/mnt/ceib/datalake/FISABIO_datalake/prueba/p0052021_reborn/derivatives/creating_adc"
t2_list = []
adc_list = []
dwi_list = []

# Creating a list of all the files in the directory starting with sub-*
subjects = [f for f in os.listdir(path) if f.startswith("sub-")]
derivatives_subs = [f for f in os.listdir(derivatives_path) if f.startswith("sub-")]

rLogger.info("Starting Data Reading")



t2w_df = pd.read_csv("/home/jaalzate/Prostate_Cancer_TFM/Data_Analysis/Tables/Train_t2w_df.csv")
#dwi_df = pd.DataFrame(dwi_list)
#adc_df = pd.DataFrame(adc_list)

rLogger.info("Data Reading Complete")
#log t2w_df shape
rLogger.info("T2w shape: %s", t2w_df.shape)

images_paths = t2w_df.copy().apply(lambda x: os.path.join(path, x['subject'], x['session'],'mim-mr','anat', x['image']), axis=1)


data = pd.DataFrame(columns=["ID", "Image"])
data["ID"] = t2w_df["subject"] + "_" + t2w_df["session"] + "_" + t2w_df["image"]
data["Image"] = images_paths.values

outPath = "Prostate_Cancer_TFM/Data_Analysis/EHRtemporalVariability/"
## Rename following files with the selected masks
outputFilepath = os.path.join(outPath, "Picai_train_radiomics_features_t2w.csv")
outputSummary = os.path.join(outPath, "Picai_train_radiomics_summary_t2w.csv")
progress_filename = os.path.join(outPath, "Picai_train_radiomics_log_t2w.txt")
params = os.path.join(outPath, "Params.yaml")


# Assuming 'radiomics' and 'rLogger' are already imported and configured elsewhere in your code.
rLogger.info("Starting Radiomics Calculation")
print("pyradiomics version:", radiomics.__version__)
print("Loading CSV")

try:
    # Use pandas to read and transpose ('.T') the input data
    # The transposition is needed so that each column represents one test case
    # This is easier for iteration over the input cases.
    flists = data.T
except Exception as e:
    print("CSV READ FAILED:", str(e))
print("Loading Done")
print("Patients:", len(flists.columns))

col_list = []
for col in flists:
    col_list.append(flists[col])

pool = Pool(processes=(cpu_count() - 1))
l_results = pool.map(calculate_radiomics, col_list)
pool.close()
pool.join()
# Merge results in one df
results = pd.DataFrame()
for result in l_results:
    results = results.join(result, how="outer")
# General info and features in two files
# Creating an only-features CSV makes R loading easier
results = results.T
info = results.filter(regex="general", axis=1).columns
print(info)
summary = results[info]
results = results.drop(info, axis=1)

print("Extraction complete T2w, writing CSVs")

results.to_csv(outputFilepath, na_rep="NaN")
print("Features CSV writing complete")
summary.to_csv(outputSummary, na_rep="NaN")
print("Summary CSV writing complete")

rLogger.info("Radiomics Calculation Complete T2w")
