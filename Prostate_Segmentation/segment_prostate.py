from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor



tmp_folder_imgs = "/mnt/ceib/datalake/FISABIO_datalake/prueba/prostate_segmentation/tmp_images"
tmp_folder_preds = "/mnt/ceib/datalake/FISABIO_datalake/prueba/prostate_segmentation/tmp_predictions"


# instantiate the nnUNetPredictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device=torch.device('cuda', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)


# initializes the network architecture, loads the checkpoint
results_folder = "/nvmescratch/ceib/Prostate/nnUnet/nnUNet_results/Dataset014_ProstateOwn/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres"
predictor.initialize_from_trained_model_folder(
    results_folder,
    use_folds=(0,1,2,3,4),
    checkpoint_name='checkpoint_best.pth',
)



# Save Segmentations
predictor.predict_from_files(tmp_folder_imgs,
                            tmp_folder_preds,
                            save_probabilities=False, overwrite=False,
                            num_processes_preprocessing=2, num_processes_segmentation_export=2,
                            folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)