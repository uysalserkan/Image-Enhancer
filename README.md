# Image-Enhancer
Enhance your low quality images with GAN.

## How can I install?

1. Clone repository.
2. Update submodule with `git submodule update --init`
3. `pip install basicsr facexlib gfpgan` install first requirements.
4. `pip install Real-ESRGAN/requirements.txt`
5. `python setup.py develop` in [Real-ESRGAN](Real-ESRGAN) folder.

## How can I use?
After installation steps you can use `enhance_images.py` script as below.
```shell
python enhance_images.py --input_path="low_images" --output_path="high_image"
```

## Run Dockerfile

1. `docker build -t image_enhancer .`
2. `docker run -it image_enhancer -v $PWD/low_images:/low_images`
