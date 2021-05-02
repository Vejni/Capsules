from src.datasets import set_seed
from src.patchwisemodel import PatchWiseModel
from src.imagewisemodels import BaseCNN, DynamicCapsules, NazeriCNN, VariationalCapsules, SRCapsules
from src.mixedmodels import VariationalMixedCapsules, EffNet
from argparse import Namespace

if __name__ == "__main__":
    set_seed()

    args_patch_wise = Namespace(
        batch_size=32,
        lr=0.001,
        epochs=100,
        augment=True,
        flip=False,
        workers=4,
        classes=4,
        input_size=[3, 512, 512],
        output_size=[3, 64, 64],
        predefined_stats=True,
        data_path="./data/ICIAR2018/patchwise_dataset",
        checkpoint_path="./models/Checkpoints/",
        name="_patchwise_"
    )

    args_img_wise = Namespace(
        lr=0.001,
        epochs=100,
        augment=True,
        flip=False,
        workers=4,
        classes=4,
        routings=3,
        lr_decay=0.9,
        lam_recon=0.392,
        pose_dim=4,
        batch_size=8,
        arch=[64,16,16,16],
        input_size=[3, 64, 64],
        output_size=[3, 64, 64],
        data_path="./data/ICIAR2018/imagewise_dataset",
        checkpoint_path="./models/Checkpoints/",
        name="_imagewise_",
        predefined_stats=False
    )
    
    # Example

    patch_wise_model = PatchWiseModel(args_patch_wise)
    patch_wise_model.train_model(args_patch_wise)
    patch_wise_model.test(args_patch_wise, voting=True)
    patch_wise_model.test_separate_classes(args_patch_wise)
    patch_wise_model.test_training(args_patch_wise)
    patch_wise_model.plot_metrics()
    patch_wise_model.save_checkpoint("./models/")
    path = patch_wise_model.save_model("./models/")

    image_wise_model = SRCapsules(args_img_wise)
    image_wise_model.train_model(args_img_wise)
    image_wise_model.test(args_img_wise, True)
    image_wise_model.test_separate_classes(args_img_wise)
    image_wise_model.test_training(args_img_wise)
    image_wise_model.plot_metrics()
    image_wise_model.save_checkpoint("./models/")
    image_wise_model.save_model("./models/")
