# 1-Lipschitz Neural Distance Fields

![banner](https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields/assets/20912374/0b92d20d-054a-4f80-8746-ccfe13fd9eae)

This repository contains the code accompanying our SGP 2024 publication _1-Lipschitz Neural Distance Fields_. Here is a link to the [project page](https://gcoiffier.github.io/publications/onelipsdf/).

## Dependencies
- numpy
- torch
- libigl 
- deel-torchlip
- mouette
- scipy
- scikit-learn
- skimage

All dependencies can be installed using pip : `pip3 install -r requirements.txt`

## Organisation of the repository

The repository has two branches 
- On the `main` branch, you will find a clean implementation of our method to be used anywhere you want.
- On the `dev` branch, you will find our experimental sandbox and scripts to reproduce the results presented in the paper.

Inside the `main` branch, you will find three families of scripts to be called:
- scripts that prepare a training dataset given some geometrical object (point cloud, surface mesh, etc.)
- scripts that run the training of a neural network on already generated datasets
- query scripts to perform geometrical queries (marching cubes, surface sampling, skeleton sampling) on a trained neural network.


## 1) Extracting a dataset

- 2D polyline or 2D mesh
- 3D polyline
- Surface mesh
- 3D point cloud

Modes are :
- `signed`: partition of training point as inside/outside to recover a signed field
- `unsigned`: partition of training points as surface/outside to recover an unsigned field
- `dist`: outputs a dataset of points with associated signed distances

## 2) Training a neural distance field

If you generated points for `foo.obj`, then call `python train_lip.py foo`

```
usage: train_lip.py [-h] [-o OUTPUT_NAME] [--unsigned] [-model {ortho,sll}]
                    [-n-layers N_LAYERS] [-n-hidden N_HIDDEN] [-ne EPOCHS]
                    [-bs BATCH_SIZE] [-tbs TEST_BATCH_SIZE]
                    [-lr LEARNING_RATE] [-lm LOSS_MARGIN] [-lmbd LOSS_LAMBDA]
                    [-cp CHECKPOINT_FREQ] [-cpu]
                    dataset

positional arguments:
  dataset               name of the dataset to train on

options:
  -h, --help            show this help message and exit
  -o OUTPUT_NAME, --output-name OUTPUT_NAME
                        custom output folder name
  --unsigned            flag for training an unsigned distance field
  -model {ortho,sll}, --model {ortho,sll}
                        Lipschitz architecture
  -n-layers N_LAYERS, --n-layers N_LAYERS
                        number of layers in the network
  -n-hidden N_HIDDEN, --n-hidden N_HIDDEN
                        size of the layers
  -ne EPOCHS, --epochs EPOCHS
                        Number of training epochs
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Train batch size
  -tbs TEST_BATCH_SIZE, --test-batch-size TEST_BATCH_SIZE
                        Test batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Adam's learning rate
  -lm LOSS_MARGIN, --loss-margin LOSS_MARGIN
                        margin m in the hKR loss
  -lmbd LOSS_LAMBDA, --loss-lambda LOSS_LAMBDA
                        lambda in the hKR loss
  -cp CHECKPOINT_FREQ, --checkpoint-freq CHECKPOINT_FREQ
                        Number of epochs between each model save
  -cpu                  force training on CPU
```

## 3) Querying the final neural field

- Marching Squares

- Marching Cubes

- Sampling a level set

- Sampling the medial axis
