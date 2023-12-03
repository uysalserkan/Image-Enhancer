# Image-Enhancer
Enhance your low quality images with GAN.

## How can I install?

1. Clone repository.
2. Update submodule with `git submodule update --init`
3. Create conda environment with `conda create -n Image-Enhancer -y`
4. Install requirements with `python -m pip intall -r requirements.txt`

## How can I use?
After installation steps you can use `enhance_images.py` script as below.
```shell
python enhance_images.py --input_path="low_images" --output_path="high_image"
```
