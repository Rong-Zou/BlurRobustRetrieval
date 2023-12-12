# Retrieval Robust to Object Motion Blur

## Table of Contents

- [Synthetic Data Generation](#synthetic-data-generation)
  - [Dependencies](#dependencies)
  - [Generation](#generation)
- [Real-world Data](#real-world-data)
- [Training and Testing](#training-and-testing)
  - [Environment](#environment)
  - [Data Preparation](#data-preparation)
  - [Training](#training)
  - [Pre-trained Model](#pre-trained-model)
  - [Testing](#testing)
- [Acknowledgments](#acknowledgments)

## Synthetic Data Generation

### Dependencies

The code is tested with Python 3.10.12.

Install the dependencies in [`requirements.txt`](/synthetic_data_gen/requirements.txt):

```bash
pip install -r requirements.txt
```

### Generation

1. Objects: download the [ShapeNetCore.v2](https://www.shapenet.org/) dataset.
2. Backgrounds: download the [LHQ](https://github.com/universome/alis) dataset, we use LHQ256. 
3. Optionally, if you want to use non-default textures for the objects, you can download other textures, e.g., [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/).
4. Change the paths in [`settings.py`](/synthetic_data_gen/settings.py) to the paths of the datasets. Also set the path to save the generated data.
5. Data preparation:
    - Generate statistics of the [ShapeNetCore.v2](https://www.shapenet.org/) dataset using [`gen_shapeNet_statistics.py`](/synthetic_data_gen/gen_shapeNet_statistics.py)
    - Remove the duplicated images in the dataset using [`clean_background_imgs.py`](/synthetic_data_gen/clean_background_imgs.py)

6. Go into the directory for storing the synthetic data. Generate images by running:
    ```bash
    python3 path/to/script/***.py
    ```
    The script can be:
    - [`render_imgs_same_bg.py`](/synthetic_data_gen/render_imgs_same_bg.py): render images captured from the same trajectory with the same background. 
    We use this script to generate the data for investigating the impact of varying levels of motion blur on retrieval performance while eliminating the influence of the background.
    - [`render_imgs_diff_bg.py`](/synthetic_data_gen/render_imgs_diff_bg.py): render images captured from the same trajectory with different backgrounds. 
    We use this script to generate the distractor set used in evaluation.
    
    **Note:** Adjust the parameters in the scripts to generate different data.

## Real-world Data

Download our real-world dataset from [this link](/real_data/link.txt).

For testing, put the dataset in the directory same as the link file.

## Training and Testing

### Environment

The code is tested with Python 3.11.4 in a Conda environment. We recommend using Anaconda to manage the dependencies.

If you don't have Conda installed, you need to first download and install it from the [official Anaconda website](https://www.anaconda.com/download#downloads).

After that, create a conda environment with the provided dependencies in [`environment.yml`](/code/environment.yml):

```bash
conda env create -f environment.yml
```

### Data Preparation

Generate statistics of the synthetic dataset using [`gen_synthetic_statistics.py`](/code/gen_synthetic_statistics.py).

### Training

Train a model by running the training script [`train.py`](/code/train.py):
  
  ```bash
  python3 train.py
  ```

  **Note:** Check the parameters in the script and change them to fit your needs. Also change the path to the synthetic dataset in [`settings.py`](/code/settings.py).

### Pre-trained Model

Download our pre-trained model from [this link](/weights/des_w1.0cls_w0.1bbox_w10.0continuous_erosion_w1.0/train_results/link.txt).

For testing, put the model in the directory same as the link file.

### Testing

Test the model by running the testing script [`test.py`](/code/test.py):
  
  ```bash
  python3 test.py
  ```

  **Note:** Modify the values of the parameters in the script to test on synthetic or real-world data under different settings.

## Acknowledgments

We thank the author of [this repo](https://github.com/rozumden/render-blur) for providing the valuable code that inspired our implementation of the synthetic data generation.
