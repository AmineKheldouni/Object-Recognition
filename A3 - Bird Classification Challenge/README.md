
## Object recognition and computer vision 2018/2019

### Assignment 3: Image classification

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
Download the training/validation/test images from [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating your model
Run the script `main.py` to train your model.

Modify `main.py`, `model.py` and `data.py` for your assignment, with an aim to make the validation score better.

- By default the images are loaded and resized to 64x64 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

#### Usage
We used a ResNet-34 architecture for birds classficiation.
You can begin by cropping the datasets (training and validation) via Mask R-CNN benchmark from Facebook's researches by running the
jupyter notebook in "maskrcnn-benchmark-master/demo/Mask_R-CNN_demo.ipynb". By plugging the joint notebook file to the right path in Facebook's repository.
Once the cropped images added to input dataset, we can run the "main.py" file by tuning the parameters (learning rate, momentum, batch size, number of epochs ...).
We used a 244x244 resize in data_transforms which slows the computation of the parameters during epochs. You can use a learning rate of 0.1 (which will be decreased each 5 epochs) and a momentum of 0.8 to ensure the stability of the convergence.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.
