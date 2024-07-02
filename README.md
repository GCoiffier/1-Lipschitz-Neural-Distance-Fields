# 1-Lipschitz Neural Distance Fields

![banner](https://github.com/GCoiffier/1-Lipschitz-Neural-Distance-Fields/assets/20912374/0b92d20d-054a-4f80-8746-ccfe13fd9eae)

This repository contains the code accompanying our SGP 2024 publication _1-Lipschitz Neural Distance Fields_. Here is a link to the [project page](https://gcoiffier.github.io/publications/onelipsdf/).

## Dependencies
- numpy
- torch 
- [libigl](https://libigl.github.io/) (implementation of signed distance for meshes and generalized winding number) 
- [deel-torchlip](https://github.com/deel-ai/deel-torchlip) (implementation of some Lipschitz neural architectures)
- [mouette](https://github.com/GCoiffier/mouette) (our mesh utility library in python)
- scipy (use of a KD-tree)
- scikit-learn (for k-nearest neighbors searches)
- skimage (for marching squares and marching cubes algorithms)

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

- `extract_dataset_2dpolyline.py` extracts points around a 2D polyline (or 2D mesh).
- `extract_dataset_3dpolyline.py` extracts points around a 3D polyline. Only datasets for unsigned distance fields can be computed.
- `extract_dataset_surface_mesh.py` extracts points around a surface mesh.
- `extract_dataset_pointcloud.py` extracts points around a 3D point cloud with normals.

These scripts can run onto different modes, depending on what type of dataset you want to generate:
- `signed`: partition of training point as inside/outside to recover a signed field
- `unsigned`: partition of training points as surface/other to recover an unsigned field
- `dist`: outputs a dataset of points with associated signed distances (not available for point cloud inputs)
- `sal`: outputs a dataset as descripted in the _Sign Agnostic Learning of Shapes From Raw Data_ paper from Atzmon and Lipman

Running these scripts will generate the corresponding datasets in the `inputs` folder. You can then use these files to train a neural distance field.

## 2) Training a neural distance field

If you generated points for `foo.obj`, then call `python train_lip.py foo`

Three training scripts are available:
- `train_lip.py` trains a Lipschitz architecture to minimize the hKR loss on a signed or unsigned dataset
- `train_fullinfo.py` trains a classical or Lipschitz architecture on a dataset of points where the ground truth distance is known. Needs a dataset generated using `dist` mode.
- `train_SALD.py` reproduces the results of the _SALD: Sign Agnostic Learning with Derivatives_ paper by Atzmon and Lipman and train a network on their proposed loss. It needs a dataset generated using `sal` mode.

These scripts output visualization and models in the `output/` folder.

## 3) Querying the final neural field

Finally, several scripts perform geometrical queries on trained neural networks:

- `reconstruct_polyline.py` runs the marching square algorithms to reconstruct a polyline of some isovalue for 2D
- `reconstruct_surface.py` runs the marching cube algorithm to reconstruct isosurfaces in 3D
- `sample_iso.py` samples a given number of points on a given isovalue.
- `sample_skeleton.py` samples the skeleton of the neural implicit shape by sample and reject depending on the neural function's gradient norm.