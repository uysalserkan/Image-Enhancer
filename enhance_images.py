"""Image enhancer service."""
import os
import glob
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
import cv2
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

from ImageDataset import ImageDataset


def init_models(args):
    """Model initialize."""
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4
    file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    model_path = os.path.join('Real-ESRGAN', 'weights', 'RealESRGAN_x4plus.pth')

    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=args.half,
        gpu_id=args.gpu
    )

    face_enhancer = GFPGANer(
        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
        upscale=args.outscale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=upsampler
    )

    return face_enhancer



def select_device() -> torch.DeviceObjType:
    """Find the GPU is exist or not.

    :returns
        `torch.DeviceObjType`: Check gpu is aviable or not.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def upscale_images(model, image_pats, args) -> None:
    """Enhance all images and save it to output folder.

    :parameters
        `model`: xyz
        `image_pats`: xyz
        `args`: xyz
    """
    for path in image_pats:
        imgname, extension = os.path.splitext(os.path.basename(path))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        _, _, output = model.enhance(
            img,
            args.has_aligned,
            args.only_center_face,
            args.paste_back
        )

        save_path = os.path.join(args.output_path, f'{imgname}{extension}')
        cv2.imwrite(save_path, output)
        print(f"{save_path} image saved.")


if __name__ == '__main__':
    parser = ArgumentParser(
        prog="Image enhancer service",
        description="Description.",
        add_help=True
    )

    parser.add_argument(
        "-i", "--input_path",
        type=str,
        required=True,
        help="Enter your images folder path."
    )

    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="Enter your images output folder path."
    )
    parser.add_argument(
        "--outscale",
        default=4,
        type=int,
    )
    parser.add_argument(
        "--pre_pad",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--tile",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--tile_pad",
        default=10,
        type=int
    )
    parser.add_argument(
        "--half",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--has_aligned",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--only_center_face",
        default=False,
        type=bool
    )
    parser.add_argument(
        "--paste_back",
        default=True,
        type=bool
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int
    )
    parser.add_argument(
        "--batch",
        default=1,
        type=int
    )

    device = select_device()

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # Model params
    model = init_models(args=args)

    # img_paths = sorted(glob.glob(os.path.join(args.input_path, '*')))

    # print("Image Paths".center(50).replace(' ', '-'))
    # print(img_paths)

    # upscale_images(model=model, image_pats=img_paths, args=args)

    image_dataset = ImageDataset(folder_path=args.input_path, output_path=args.output_path, model=model)
    dataloader = DataLoader(dataset=image_dataset, batch_size=args.batch, shuffle=True)
    for _batch in dataloader:
        print(f"{_batch}".center(50))

    print("Upscaling done".center(50).replace(' ', '-'))

