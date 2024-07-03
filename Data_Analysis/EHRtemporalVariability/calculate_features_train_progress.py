import pandas as pd
import numpy as np
import os
import json
from numpy import std, mean
from radiomics import featureextractor
import SimpleITK as sitk
from multiprocessing import cpu_count, Pool
import radiomics 
import logging
import pickle

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

def calculate_radiomics(case, params): 
    rLogger.info("Processing Patient: %s", case["ID"])
    if os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else: 
        settings = {
            "binWidth": 25,
            "resampledPixelSpacing": [0.5, 0.5, 3],
            "interpolator": sitk.sitkBSpline,
            "correctMask": True,
            "normalize": True
        }
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)

    image_path = case["Image"]
    image = sitk.ReadImage(image_path)

    mask, _ = delete_black_slices(image, threshold=0.5)
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
            

    values = np.array([1])
    patient = []
    for index, label in enumerate(values, start=1):
        label = int(label)
        rLogger.info("Processing Patient %s (Image: %s, Label: %s)", case["ID"], case["Image"], label)
        try:
            result = pd.Series(extractor.execute(image, mask, label))
        except Exception:
            rLogger.error("FEATURE EXTRACTION FAILED:", exc_info=True)
            rLogger.error("Skipping Patient %s", case["ID"])
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

    return case["ID"], patient

def save_intermediate_results(results, processed_patients, outputFilepath, progress_filepath):
    results.to_csv(outputFilepath, na_rep="NaN")
    with open(progress_filepath, 'wb') as f:
        pickle.dump(processed_patients, f)

def load_progress(progress_filepath):
    if os.path.exists(progress_filepath):
        with open(progress_filepath, 'rb') as f:
            return pickle.load(f)
    return []

# Path to the data
path = "/mnt/ceib/datalake/FISABIO_datalake/prueba/p0052021_reborn"
derivatives_path = "/mnt/ceib/datalake/FISABIO_datalake/prueba/p0052021_reborn/derivatives/creating_adc"
t2_list = []
adc_list = []
dwi_list = []

subjects = [f for f in os.listdir(path) if f.startswith("sub-")]
derivatives_subs = [f for f in os.listdir(derivatives_path) if f.startswith("sub-")]

rLogger.info("Starting Data Reading")

t2w_df = pd.read_csv("/home/jaalzate/Prostate_Cancer_TFM/Data_Analysis/Tables/Train_t2w_df.csv")

rLogger.info("Data Reading Complete")
rLogger.info("T2w shape: %s", t2w_df.shape)

images_paths = t2w_df.copy().apply(lambda x: os.path.join(path, x['subject'], x['session'],'mim-mr','anat', x['image']), axis=1)

data = pd.DataFrame(columns=["ID", "Image"])
data["ID"] = t2w_df["subject"] + "_" + t2w_df["session"] + "_" + t2w_df["image"]
data["Image"] = images_paths.values

outPath = "/home/jaalzate/Prostate_Cancer_TFM/Data_Analysis/EHRtemporalVariability/"
outputFilepath = os.path.join(outPath, "Picai_train_radiomics_features_t2w.csv")
outputSummary = os.path.join(outPath, "Picai_train_radiomics_summary_t2w.csv")
progress_filepath = os.path.join(outPath, "Picai_train_radiomics_progress.pkl")
params = os.path.join(outPath, "Params.yaml")

rLogger.info("Starting Radiomics Calculation")
print("pyradiomics version:", radiomics.__version__)
print("Loading CSV")

try:
    flists = data.T
except Exception as e:
    print("CSV READ FAILED:", str(e))
print("Loading Done")
print("Patients:", len(flists.columns))

processed_patients = load_progress(progress_filepath)
processed_ids = set(processed_patients)
data = data[~data['ID'].isin(processed_ids)]

col_list = []
for col in flists:
    if flists[col]['ID'] not in processed_ids:
        col_list.append(flists[col])

results = pd.DataFrame()
n = 10  # Save every n patients

for i in range(0, len(col_list), n):
    batch = col_list[i:i+n]
    pool = Pool(processes=(cpu_count() - 1))
    l_results = pool.starmap(calculate_radiomics, [(case, params) for case in batch])
    pool.close()
    pool.join()
    
    batch_results = pd.DataFrame()
    for patient_id, result in l_results:
        batch_results = batch_results.join(result, how="outer")
        processed_patients.append(patient_id)
    
    results = results.join(batch_results, how="outer")
    save_intermediate_results(results, processed_patients, outputFilepath, progress_filepath)

results = results.T
info = results.filter(regex="general", axis=1).columns
summary = results[info]
results = results.drop(info, axis=1)

results.to_csv(outputFilepath, na_rep="NaN")
summary.to_csv(outputSummary, na_rep="NaN")

rLogger.info("Radiomics Calculation Complete T2w")
