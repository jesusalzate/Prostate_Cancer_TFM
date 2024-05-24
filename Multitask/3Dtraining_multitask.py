from json import dump
from os.path import join
from re import search
from time import strftime

import nibabel as nib
import torch
from monai import transforms
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.networks.nets import EfficientNetBN
from numpy import unique
from pandas import DataFrame, read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch import device as pytorch_device
from torch import tensor
from torch.cuda import is_available
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.optim import Adadelta, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, ConfusionMatrix, Recall, Specificity

from bimcv_aikit.loss_functions.multitask.loss_multitask import seg_and_class_loss_multitask
from bimcv_aikit.metrics.multitask.metrics_segmentation_multitask import metrics_segmentation_multitask
#from bimcv_aikit.monai.models.SwinUnetr_multitask import SwinUnetr_multitask
from bimcv_aikit.monai.transforms import ConcatLabels_Multitask
from bimcv_aikit.pytorch.metrics.metrics_multitask import (
    metrics_classification_multitask,
)
from bimcv_aikit.pytorch.model_complexity import (
    calculate_model_complexity,
    count_parameters_pytorch,
)
from bimcv_aikit.pytorch.training_multitask import evaluate, train
from model import SwinUnetr_multitask_post
# ## PyTorch device configuration


def extract_data_from_df_multitask(path: str, sep: str):
    df = read_csv(path, sep=sep)

    format_load = "cropped"  #'nifti', mha, cropped

    df["depth"] = df["filepath_t2w_" + format_load].apply(
        lambda path_file: nib.load(path_file).shape[0]
    )
    df["heigth"] = df["filepath_t2w_" + format_load].apply(
        lambda path_file: nib.load(path_file).shape[1]
    )
    df["weigth"] = df["filepath_t2w_" + format_load].apply(
        lambda path_file: nib.load(path_file).shape[2]
    )
    df = df[(df["heigth"] != 0) & (df["depth"] != 0)]
    df = df[(df["heigth"] > 96) & (df["depth"] > 96)]

    data_picai = df[df["filepath_t2w_" + format_load].notna()].reset_index()

    data_picai_human = data_picai[data_picai["human_labeled"] == 1]
    data_picai.drop(data_picai_human.index, inplace=True)

    data_picai = data_picai[
        [
            "filepath_t2w_" + format_load,
            "filepath_adc_" + format_load,
            "filepath_hbv_" + format_load,
            "filepath_labelAI_" + format_load,
            "filepath_seg_zones_" + format_load,
            "label",
            "partition",
        ]
    ]
    data_picai_human = data_picai_human[
        [
            "filepath_t2w_" + format_load,
            "filepath_adc_" + format_load,
            "filepath_hbv_" + format_load,
            "filepath_label_" + format_load,
            "filepath_seg_zones_" + format_load,
            "label",
            "partition",
        ]
    ]

    train_picai = data_picai[data_picai["partition"] == "tr"]
    val_picai = data_picai[data_picai["partition"] == "dev"]
    test_picai = data_picai[data_picai["partition"] == "test"]

    train_picai_human = data_picai_human[data_picai_human["partition"] == "tr"]
    val_picai_human = data_picai_human[data_picai_human["partition"] == "dev"]
    test_picai_human = data_picai_human[data_picai_human["partition"] == "test"]

    train_df = DataFrame(
        {
            "t2w": list(train_picai["filepath_t2w_" + format_load].values)
            + list(train_picai_human["filepath_t2w_" + format_load].values),
            "adc": list(train_picai["filepath_adc_" + format_load].values)
            + list(train_picai_human["filepath_adc_" + format_load].values),
            "dwi": list(train_picai["filepath_hbv_" + format_load].values)
            + list(train_picai_human["filepath_hbv_" + format_load].values),
            "zones": list(train_picai["filepath_seg_zones_" + format_load].values)
            + list(train_picai_human["filepath_seg_zones_" + format_load].values),
            "label_seg": list(train_picai["filepath_labelAI_" + format_load].values)
            + list(train_picai_human["filepath_label_" + format_load].values),
            "label_class": list(train_picai["label"].values)
            + list(train_picai_human["label"].values),
        }
    )

    val_df = DataFrame(
        {
            "t2w": list(val_picai["filepath_t2w_" + format_load].values)
            + list(val_picai_human["filepath_t2w_" + format_load].values),
            "adc": list(val_picai["filepath_adc_" + format_load].values)
            + list(val_picai_human["filepath_adc_" + format_load].values),
            "dwi": list(val_picai["filepath_hbv_" + format_load].values)
            + list(val_picai_human["filepath_hbv_" + format_load].values),
            "zones": list(val_picai["filepath_seg_zones_" + format_load].values)
            + list(val_picai_human["filepath_seg_zones_" + format_load].values),
            "label_seg": list(val_picai["filepath_labelAI_" + format_load].values)
            + list(val_picai_human["filepath_label_" + format_load].values),
            "label_class": list(val_picai["label"].values)
            + list(val_picai_human["label"].values),
        }
    )

    test_df = DataFrame(
        {
            "t2w": list(test_picai["filepath_t2w_" + format_load].values)
            + list(test_picai_human["filepath_t2w_" + format_load].values),
            "adc": list(test_picai["filepath_adc_" + format_load].values)
            + list(test_picai_human["filepath_adc_" + format_load].values),
            "dwi": list(test_picai["filepath_hbv_" + format_load].values)
            + list(test_picai_human["filepath_hbv_" + format_load].values),
            "zones": list(test_picai["filepath_seg_zones_" + format_load].values)
            + list(test_picai_human["filepath_seg_zones_" + format_load].values),
            "label_seg": list(test_picai["filepath_labelAI_" + format_load].values)
            + list(test_picai_human["filepath_label_" + format_load].values),
            "label_class": list(test_picai["label"].values)
            + list(test_picai_human["label"].values),
        }
    )

    train_data = [
        {
            "t2": t2,
            "adc": adc,
            "dwi": dwi,
            "label_class": label_class,
            "zones": zone,
            "label_seg": label_seg,
        }
        for t2, adc, dwi, label_class, zone, label_seg in zip(
            train_df["t2w"].values,
            train_df["adc"].values,
            train_df["dwi"].values,
            one_hot(as_tensor(train_df["label_class"].values)).float(),
            train_df["zones"].values,
            train_df["label_seg"].values,
        )
    ]
    val_data = [
        {
            "t2": t2,
            "adc": adc,
            "dwi": dwi,
            "label_class": label_class,
            "zones": zone,
            "label_seg": label_seg,
        }
        for t2, adc, dwi, label_class, zone, label_seg in zip(
            val_df["t2w"].values,
            val_df["adc"].values,
            val_df["dwi"].values,
            one_hot(as_tensor(val_df["label_class"].values)).float(),
            val_df["zones"].values,
            val_df["label_seg"].values,
        )
    ]

    test_data = [
        {
            "t2": t2,
            "adc": adc,
            "dwi": dwi,
            "label_class": label_class,
            "zones": zone,
            "label_seg": label_seg,
        }
        for t2, adc, dwi, label_class, zone, label_seg in zip(
            test_df["t2w"].values,
            test_df["adc"].values,
            test_df["dwi"].values,
            one_hot(as_tensor(test_df["label_class"].values)).float(),
            test_df["zones"].values,
            test_df["label_seg"].values,
        )
    ]

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique(train_df["label_class"].values),
        y=train_df["label_class"].values,
    )
    print(f"Train images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Test images: {len(test_data)}")
    return train_data, val_data, test_data, class_weights


pin_memory = is_available()
device = pytorch_device("cuda" if pin_memory else "cpu")
print(f"Working on device: {device}")

# # Data

path_to_table = "/home/jaalzate/Tartaglia/Prostate_Tartaglia/codes/partition_1.csv"
classes = ["no_csPCa", "csPCa"]
map_labels = None
train_data, val_data, test_data, class_weights = extract_data_from_df_multitask(
    path_to_table, sep=","
)


img_columns = ["t2", "adc", "dwi"]
label_column = ["label_seg"]
mode = ["bilinear", "nearest"]
# ## DataLoaders and pre-processing

input_shape = (128, 128, 32)
prob = 0.1


train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(
            keys=img_columns + label_column + ["zones"],
            reader="NibabelReader",
            image_only=True,
        ),
        transforms.AsDiscreted(
            keys=label_column, threshold=1
        ),  # Convert values greater than 1 to 1
        transforms.EnsureChannelFirstd(keys=img_columns + label_column + ["zones"]),
        transforms.AsDiscreted(keys="zones", argmax=False, to_onehot=3),
        transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
        transforms.Resized(
            keys=img_columns + label_column + ["zones"],
            spatial_size=input_shape,
            mode=("trilinear", "trilinear", "trilinear", "nearest", "nearest"),
        ),  # SAMUNETR: Reshape to have the same dimension
        transforms.ResampleToMatchd(
            keys=["adc", "dwi", "zones", "label_seg"],
            key_dst="t2",
            mode=("bilinear", "bilinear", "nearest", "nearest"),
        ),  # Resample images to t2 dimension
        transforms.ScaleIntensityd(keys=img_columns, minv=0.0, maxv=1.0),
        transforms.NormalizeIntensityd(keys=img_columns),
        transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
        # transforms.AsDiscreted(keys=label_column,to_onehot=2),
        ConcatLabels_Multitask(keys=label_column + ["label_class"], name="label"),
        # transforms.ToTensord(keys='label')
        # transforms.ConcatItemsd(keys=label_column+['label_class'], name='label', dim=0),
        # transforms.RandSpatialCropSamplesd(keys=['image','label'],roi_size=[96,96,-1],num_samples=8,random_size=False),#For the other models
        # transforms.RandRotate90d(keys=['image','label'],spatial_axes=[0,1],prob=prob),
        # transforms.RandZoomd(keys=['image','label'],min_zoom=0.9,max_zoom=1.1,mode=['area' if x == 'bilinear' else x for x in mode],prob=prob),
        # transforms.RandGaussianNoised(keys=["image"],mean=0.1,std=0.25,prob=prob),
        # transforms.RandShiftIntensityd(keys=["image"],offsets=0.2,prob=prob),
        # transforms.RandGaussianSharpend(keys=['image'],sigma1_x=[0.5, 1.0],sigma1_y=[0.5, 1.0],sigma1_z=[0.5, 1.0],sigma2_x=[0.5, 1.0],sigma2_y=[0.5, 1.0],sigma2_z=[0.5, 1.0],alpha=[10.0,30.0],prob=prob),
        # transforms.RandAdjustContrastd(keys=['image'],gamma=2.0,prob=prob),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(
            keys=img_columns + label_column + ["zones"],
            reader="NibabelReader",
            image_only=True,
        ),
        transforms.AsDiscreted(
            keys=label_column, threshold=1
        ),  # Convert values greater than 1 to 1
        transforms.EnsureChannelFirstd(keys=img_columns + label_column + ["zones"]),
        transforms.AsDiscreted(keys="zones", argmax=False, to_onehot=3),
        transforms.LabelToMaskd(keys="zones", select_labels=[1, 2]),
        transforms.Resized(
            keys=img_columns + label_column + ["zones"],
            spatial_size=input_shape,
            mode=("trilinear", "trilinear", "trilinear", "nearest", "nearest"),
        ),  # SAMUNETR: Reshape to have the same dimension
        transforms.ResampleToMatchd(
            keys=["adc", "dwi", "zones", "label_seg"],
            key_dst="t2",
            mode=("bilinear", "bilinear", "nearest", "nearest"),
        ),  # Resample images to t2 dimension
        transforms.ScaleIntensityd(keys=img_columns, minv=0.0, maxv=255.0),
        transforms.NormalizeIntensityd(keys=img_columns),
        transforms.ConcatItemsd(keys=img_columns + ["zones"], name="image", dim=0),
        # transforms.AsDiscreted(keys=label_column,to_onehot=2),
        ConcatLabels_Multitask(keys=label_column + ["label_class"], name="label"),
        # transforms.ToTensord(keys='label')
        # transforms.ConcatItemsd(keys=label_column+['label_class'], name='label', dim=0),
    ]
)

batch_size = 4

train_ds = CacheDataset(data=train_data, transform=train_transforms, num_workers=7)
train_loader = DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, drop_last=True
)

val_ds = CacheDataset(data=val_data, transform=val_transforms, num_workers=7)
val_loader = DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, drop_last=True
)

# first = next(iter(train_loader))
# print(first["image_meta_dict"]["filename_or_obj"])
# image = first["image"]
# print(image.shape)
# label = first["label"]
# print(label)

# from monai.visualize import matshow3d
# from matplotlib.pyplot import show
# %matplotlib inline
# matshow3d(volume=image, frame_dim=-1, cmap="gray", show=True)
# show()


n_classes = len(class_weights)
# model = EfficientNetBN(model_name="efficientnet-b7", spatial_dims=3, in_channels=5, num_classes=n_classes, pretrained=True, progress=False).to(
#     device
# )
weigths = "/home/jaalzate/Tartaglia/Prostate_Tartaglia/codes/New_training_method/model_swinvit.pt"

model = SwinUnetr_multitask_post(
    n_classes=2,
    img_size=input_shape,
    in_channels=3,
    seg_channels=2,
    pretrained_weights=weigths,
).to(device)


# loss_function = CrossEntropyLoss(weight=tensor(class_weights).to(device))

loss_function_segmentation = DiceFocalLoss(
    to_onehot_y=True, sigmoid=False, softmax=True, include_background=True
)
loss_function_classification = CrossEntropyLoss(weight=tensor(class_weights).to(device))

loss_function = seg_and_class_loss_multitask(
    segmentation_loss=loss_function_segmentation,
    classification_loss=loss_function_classification,
)

optimizer = Adam(model.parameters(), lr=1e-5, amsgrad=True)
# optimizer = Adadelta(model.parameters(), lr=1e-6, rho=0.95, eps=1e-07)
scheduler = ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, verbose=True
)


experiment_name = f"{model._get_name()}_{strftime('%d-%b-%Y-%H:%M:%S')}"
log_dir = join(
    "/home/jaalzate/Tartaglia/Prostate_Tartaglia/codes/logs/Classification/3D",
    experiment_name,
)
config = {
    "checkpoint_interval": 10,
    "device": device,
    # "early_stopping":{"patience":10,"tolerance":1e-4},
    "scheduler": scheduler,
    "epochs": 100,
    "experiment_name": experiment_name,
    "loss_function": loss_function,
    "metrics": {
        "accuracy": metrics_classification_multitask(
            original_metric=Accuracy(
                task="multiclass", average="weighted", num_classes=n_classes
            )
        ),
        "specificity": metrics_classification_multitask(
            original_metric=Specificity(
                task="multiclass", average="weighted", num_classes=n_classes
            )
        ),
        "recall": metrics_classification_multitask(
            original_metric=Recall(
                task="multiclass", average="weighted", num_classes=n_classes
            )
        ),
        "dice_score": metrics_segmentation_multitask(
            original_metric=DiceMetric(
                include_background=False,
                reduction="mean",
                get_not_nans=False,
                ignore_empty=True,
            )
        ),
        "iou": metrics_segmentation_multitask(
            original_metric=MeanIoU(
                include_background=False,
                reduction="mean",
                get_not_nans=False,
                ignore_empty=True,
            )
        ),
    },
    "optimizer": optimizer,
    "save_weights_dir": log_dir,
    "tensorboard_writer": SummaryWriter(log_dir=log_dir),
    "validation_interval": 1,
    "verbose": True,
}

model = train(model, train_loader, val_loader, config=config)

metrics = {
    "accuracy": metrics_classification_multitask(
        original_metric=Accuracy(
            task="multiclass", average="weighted", num_classes=n_classes
        )
    ),
    "specificity": metrics_classification_multitask(
        original_metric=Specificity(
            task="multiclass", average="weighted", num_classes=n_classes
        )
    ),
    "recall": metrics_classification_multitask(
        original_metric=Recall(
            task="multiclass", average="weighted", num_classes=n_classes
        )
    ),
    "dice_score": metrics_segmentation_multitask(
        original_metric=DiceMetric(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            ignore_empty=True,
        )
    ),
    "iou": metrics_segmentation_multitask(
        original_metric=MeanIoU(
            include_background=False,
            reduction="mean",
            get_not_nans=False,
            ignore_empty=True,
        )
    ),
}

weights_path = join(log_dir, experiment_name + ".pth")

train_predictions, train_results = evaluate(
    model, train_loader, metrics, weights=weights_path, device=device
)
val_predictions, val_results = evaluate(
    model, val_loader, metrics, weights=weights_path, device=device
)

del train_ds, val_ds

test_ds = CacheDataset(data=test_data, transform=val_transforms, num_workers=7)
test_loader = DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, drop_last=True
)

test_predictions, test_results = evaluate(
    model, test_loader, metrics, weights=weights_path, device=device
)

experiment_summary = {
    "Experiment Name": experiment_name,
    "Log": log_dir,
    "Model": model._get_name(),
    "Model Trainable Parameters": count_parameters_pytorch(model),
    "Model Flops": calculate_model_complexity(model, (5, *input_shape)),
    "Num Classes": n_classes,
    "Spatial Dims": f"{len(input_shape)}D",
    "Classes": classes if classes else list(map_labels.keys()),
    "Epochs": config["epochs"],
    "Loss Function": loss_function._get_name(),
    "Optimizer": optimizer.__str__().replace("\n    ", ", ").replace("\n", ""),
    "Train Transforms": [
        search(r"[\d]?[A-Z]\w+", transform.__str__())[0]
        for transform in train_loader.dataset.transform.transforms
    ],
    "Val Transforms": [
        search(r"[\d]?[A-Z]\w+", transform.__str__())[0]
        for transform in val_loader.dataset.transform.transforms
    ],
    "Results": {
        "Train Predictions": train_predictions,
        "Train Metrics": train_results,
        "Val Predictions": val_predictions,
        "Val Metrics": val_results,
        "Test Predictions": test_predictions,
        "Test Metrics": test_results,
    },
}

with open(f"{log_dir}/summary.json", "w") as json_file:
    dump(experiment_summary, json_file, ensure_ascii=False, indent=4)
