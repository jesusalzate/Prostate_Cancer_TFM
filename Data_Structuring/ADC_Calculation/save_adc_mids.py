import json
import os

import matplotlib.pyplot as plt
import monai
import pandas as pd
from tqdm import tqdm
from monai.data import MetaTensor
import nibabel as nib
import numpy as np
from scipy.stats import linregress
from multiprocessing import Pool
from bimcv_aikit.medical_imaging.mri.calculate_adc import get_save_adc

tqdm.pandas()


#using argparse to get the path of the csv file

import argparse

parser = argparse.ArgumentParser(description='Data Structure for Prostate Cancer Dataset')
parser.add_argument('--csv_path', type=str, help='Path to the csv file')
parser.add_argument('--source_path', type=str, help='Path to the source folder')
parser.add_argument('--exp_name', type=str, default='creating_adc', help='Experiment name')

args = parser.parse_args()


#new_df = pd.read_csv("/home/jaalzate/Tartaglia/Prostate_Tartaglia/New_Dataset/validation_dataset.csv")
new_df = pd.read_csv(args.csv_path)

#source_path = "/mnt/ceib/datalake/FISABIO_datalake/p0042021/"
source_path = args.source_path


dwi_protocols = {
    "ax dif  b1000": (1000, 0),
    "ax dif b1500": (1500, 0),
    "ax dif 2000": (2000, 0),
    "Ax DWI b50-800 PELVIS": (800, 50),
    "ax dif b1000": (1000, 0),
    "Ax DWI b2000": (2000, 0),
    "AX DWI B1000": (1000, 0),
    "AX DWI B800": (800, 0),
    "Ax DWI b1500": (1500, 0),
    "DWI_EPI_SPAIR_TRA B2000": (0, 2000),  # Pattern change
    "ax dif b1000": (1000, 0),
    "ax dif  b1500": (1500, 0),
    # "DW_Synthetic: AXI PELVIS DWI B50-800": None,  # Single channel
    "AXI PELVIS DWI B50-800": (50, 800),  # Pattern change
    "ax dif b1800": (1800, 0),
    "AX DWI b1500": (1500, 0),
    "Ax DWI b1500 FOV22 PRUEBA": (1500, 0),
    "ep2d_diff_tra_b0_b1000": (0, 1000),  # Pattern change
    "ax DWI b2000": (2000, 0),
}


# Define Experiment name

exp_name = "creating_adc"


for i, row in tqdm(new_df.iterrows()):
    if row.can_calculate_adc:
        dwi_path = os.path.join(
            source_path, row.subject, row.session, "mim-mr", row.modality, row.image
        )

        dwi_json = dwi_path.replace(".nii.gz", ".json")
        b_values = dwi_protocols[row.protocol_association]

        # adc_map_name='_'.join(row.image.split('_')[:-1])+'_adc.nii.gz'
        adc_map_name = row.image.replace(
            "dwi.nii.gz", "mod-dwi_desc-calcADC_ADC.nii.gz"
        )
        adc_json_name = row.image.replace(
            ".nii.gz", ".json"
        )

        adc_path = os.path.join(
            source_path,
            "derivatives",
            exp_name,
            row.subject,
            row.session,
            "mim-mr",
            "dwi",
        )

        if os.path.exists(os.path.join(adc_path, adc_map_name)):
            print("ADC map already exists: "+adc_map_name)
        else:
            adc_map, dwi = get_save_adc(dwi_path, b_values)

            if dwi.ndim > 3:
                # create a meta tensor from the numpy array

                meta = dwi.meta
                meta["filename_or_obj"] = os.path.join(adc_path, adc_map_name)
                meta["b_values"] = b_values

                adc_map = MetaTensor(adc_map, meta=meta)

                monai.transforms.SaveImage(
                    output_dir=adc_path,
                    output_postfix="",
                    output_ext=".nii.gz",
                    resample=False,
                    scale=None,
                    squeeze_end_dims=True,
                    data_root_dir="",
                    separate_folder=False,
                    print_log=True,
                    channel_dim=None,
                )(adc_map)
        
        # Create json file
        if os.path.exists(os.path.join(adc_path, adc_json_name)):
            print("Json file already exists: " + adc_json_name)
        else:
            if os.path.exists(os.path.join(adc_path, adc_map_name)):
                adc_map = nib.load(os.path.join(adc_path, adc_map_name))
                print("Creating json file")
                #Read DWI json file
                with open(dwi_json) as f:
                    dwi_meta = json.load(f)
                
                adc_dict = {
                    "AccessionNumber": dwi_meta["AccessionNumber"],
                    "AcquisitionDate": dwi_meta["AcquisitionDate"],
                    "AcquisitionMatrix": dwi_meta["AcquisitionMatrix"],
                    "AcquisitionTime": dwi_meta["AcquisitionTime"],
                    "BitsAllocated": dwi_meta["BitsAllocated"],
                    "BitsStored": dwi_meta["BitsStored"],
                    "BodyPartExamined": "Prostate" if "BodyPartExamined" not in dwi_meta.keys() else dwi_meta["BodyPartExamined"],
                    "CardiacNumberOfImages": 0 if "CardiacNumberOfImages" not in dwi_meta.keys() else dwi_meta["CardiacNumberOfImages"],
                    "Columns": dwi_meta["Columns"],
                    "ContentDate": dwi_meta["ContentDate"],
                    "ContentTime": dwi_meta["ContentTime"],
                    "DeidentificationMethod": dwi_meta["DeidentificationMethod"],
                    "DeidentificationMethodCodeSequence": dwi_meta["DeidentificationMethodCodeSequence"],
                    "EchoNumbers": dwi_meta["EchoNumbers"],
                    "EchoTime": dwi_meta["EchoTime"],
                    "EchoTrainLength": dwi_meta["EchoTrainLength"],
                    "FlipAngle": dwi_meta["FlipAngle"],
                    "FrameOfReferenceUID": dwi_meta["FrameOfReferenceUID"],
                    "HighBit": dwi_meta["HighBit"],
                    "ImageOrientationPatient": dwi_meta["ImageOrientationPatient"],
                    "ImagePositionPatient": dwi_meta["ImagePositionPatient"],
                    "ImageType": [
                        "DERIVED",
                        "PRIMARY",
                        "ADC",
                        "ADC"
                    ],
                    "ImagesinAcquisition": 0 if "ImagesinAcquisition" not in dwi_meta.keys() else dwi_meta["ImagesinAcquisition"],
                    "ImagingFrequency": dwi_meta["ImagingFrequency"],
                    "InPlanePhaseEncodingDirection": dwi_meta["InPlanePhaseEncodingDirection"],
                    "InstanceNumber": 0,
                    "InversionTime": None if "InversionTime" not in dwi_meta.keys() else dwi_meta["InversionTime"],
                    "LargestImagePixelValue": max(adc_map.get_fdata().flatten()),
                    "Laterality": "" if "Laterality" not in dwi_meta.keys() else dwi_meta["Laterality"],
                    "LongitudinalTemporalInformationModified": dwi_meta["LongitudinalTemporalInformationModified"],
                    "MRAcquisitionType": dwi_meta["MRAcquisitionType"],
                    "MagneticFieldStrength": dwi_meta["MagneticFieldStrength"],
                    "Manufacturer": dwi_meta["Manufacturer"],
                    "ManufacturerModelName": dwi_meta["ManufacturerModelName"],
                    "Modality": dwi_meta["Modality"],
                    "PatientAge": dwi_meta["PatientAge"],
                    "PatientBirthDate": dwi_meta["PatientBirthDate"],
                    "PatientComments": dwi_meta["PatientComments"],
                    "PatientID": dwi_meta["PatientID"],
                    "PatientName": dwi_meta["PatientName"],
                    "PatientPosition": dwi_meta["PatientPosition"],
                    "PatientSex": dwi_meta["PatientSex"],
                    "PatientWeight": dwi_meta["PatientWeight"],
                    "PhotometricInterpretation": dwi_meta["PhotometricInterpretation"],
                    "PixelBandwidth": dwi_meta["PixelBandwidth"],
                    "PixelRepresentation": dwi_meta["PixelRepresentation"],
                    "PixelSpacing": dwi_meta["PixelSpacing"],
                    "PositionReferenceIndicator": "" if "PositionReferenceIndicator" not in dwi_meta.keys() else dwi_meta["PositionReferenceIndicator"],
                    "ReceiveCoilName": "HD Cardiac" if "ReceiveCoilName" not in dwi_meta.keys() else dwi_meta["ReceiveCoilName"],
                    "ReconstructionDiameter": 0 if "ReconstructionDiameter" not in dwi_meta.keys() else dwi_meta["ReconstructionDiameter"],
                    "RepetitionTime": dwi_meta["RepetitionTime"],
                    "Rows": dwi_meta["Rows"],
                    "SOPClassUID": dwi_meta["SOPClassUID"],
                    "SOPInstanceUID": dwi_meta["SOPInstanceUID"],
                    "SamplesPerPixel": dwi_meta["SamplesPerPixel"],
                    "ScanOptions": dwi_meta["ScanOptions"],
                    "ScanningSequence": dwi_meta["ScanningSequence"],
                    "SequenceVariant": dwi_meta["SequenceVariant"],
                    "SeriesDate": dwi_meta["SeriesDate"],
                    "SeriesDescription": "Apparent Diffusion Coefficient (mm2/s)",
                    "SeriesInstanceUID": dwi_meta["SeriesInstanceUID"],
                    "SeriesNumber": dwi_meta["SeriesNumber"]*100,
                    "SeriesTime": dwi_meta["SeriesTime"],
                    "SliceLocation": 0,
                    "SliceThickness": dwi_meta["SliceThickness"],
                    "SmallestImagePixelValue": min(adc_map.get_fdata().flatten()),
                    "SoftwareVersions": dwi_meta["SoftwareVersions"],
                    "SpacingBetweenSlices": dwi_meta["SpacingBetweenSlices"],
                    "SpecificCharacterSet": dwi_meta["SpecificCharacterSet"],
                    "StudyDate": dwi_meta["StudyDate"],
                    "StudyDescription": dwi_meta["StudyDescription"],
                    "StudyID": dwi_meta["StudyID"],
                    "StudyInstanceUID": dwi_meta["StudyInstanceUID"],
                    "StudyTime": dwi_meta["StudyTime"],
                    "TriggerTime": "NaN",
                    "WindowCenter": 0,
                    "WindowWidth": 0,
                    "bvalues": b_values,
                }

                json_object = json.dumps(adc_dict, indent=4)

                # write the JSON string to a file
                with open(
                    os.path.join(
                        adc_path,
                        row.image.replace(
                            "dwi.nii.gz", "mod-dwi_desc-calcADC_ADC.json"
                        ),
                    ),
                    "w",
                ) as outfile:
                    outfile.write(json_object)
