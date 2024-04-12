# 1-Lipschitz Neural Distance Fields


## Dependencies
- numpy
- torch
- libigl
- deel-torchlip
- mouette
- scipy
- scikit-learn
- skimage

## How to run

#### Extracting a dataset

- 2D polyline or 2D mesh
- 3D polyline
- Surface mesh
- 3D point cloud

Modes are :
- `signed`: partition of training point as inside/outside to recover a signed field
- `unsigned`: partition of training points as surface/outside to recover an unsigned field
- `dist`: outputs a dataset of points with associated signed distances

#### Training a neural distance field

If you generated points for `foo.obj`, then call:

`python train_lip.py foo`
Options are:
  `-o OUTPUT_NAME, --output-name OUTPUT_NAME`
        Custom output name

  `--unsigned`
        To train an unsigned distance field (needed to be compatible with unsigned datasets)

  `-model {ortho,sll}, --model {ortho,sll}`
        Lipschitz architecture. Default is `sll`. `ortho` is the architecture described by Anil et al.
  
  `-n-layers N_LAYERS, --n-layers N_LAYERS`
        Number of Lipschitz layers in the network

  `-n-hidden N_HIDDEN, --n-hidden N_HIDDEN`
        Size of the Lipschitz layers in the network

  `-ne EPOCHS, --epochs EPOCHS`
        Number of training epochs

  `-bs BATCH_SIZE, --batch-size BATCH_SIZE`
        Training batch size

  `-tbs TEST_BATCH_SIZE, --test-batch-size TEST_BATCH_SIZE`
        Testing batch size
  
  `-lr LEARNING_RATE, --learning-rate LEARNING_RATE`
        Adam's learning rate
  
  `-lm LOSS_MARGIN, --loss-margin LOSS_MARGIN`
        Margin parameter m in the hKR loss

  `-lmbd LOSS_LAMBDA, --loss-lambda LOSS_LAMBDA`
        Parameter lambda in the hKR loss

  `-cp CHECKPOINT_FREQ, --checkpoint-freq CHECKPOINT_FREQ`
        Frequency of model checkpoints

  `-cpu`
        Force training on the CPU


#### Querying the final neural field

- Marching Squares

- Marching Cubes

- Sampling a level set

- Sampling the medial axis