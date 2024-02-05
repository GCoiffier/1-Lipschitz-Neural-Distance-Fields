import os
import argparse
from types import SimpleNamespace

import mouette as M

from common.config import Config
from common.dataset import PointCloudDataset_FullInfo
from common.model import *
from common.callback import *
from common.visualize import point_cloud_from_tensors
from common.training import Trainer
from common.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="")
    parser.add_argument("-ne", "--epochs", type=int, default=10, help="Number of epochs per iteration")
    parser.add_argument('-bs',"--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("-tbs", "--test-batch-size", type = int, default = 5000, help="Batch size on test set")
    parser.add_argument("-cp", "--checkpoint-freq", type=int, default=10)
    parser.add_argument("-a", "--attach-weight", type=float, default=0.)
    parser.add_argument("-b", "--normal-weight", type=float, default=0.)
    parser.add_argument("-cpu", action="store_true")
    args = parser.parse_args()

    #### Config ####
    config = Config(
        dim = 3,
        device = get_device(args.cpu),
        n_epochs = args.epochs,
        checkpoint_freq = args.checkpoint_freq,
        batch_size = args.batch_size,
        test_batch_size = args.test_batch_size,
        optimizer = "adam",
        learning_rate = 1e-4,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load dataset ####
    dataset = PointCloudDataset_FullInfo(args.dataset, config)
    plot_domain = dataset.object_BB
    plot_domain.pad(0.5, 0.5, 0.5)

    #### Create model and setup trainer
    # archi = [(3,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1)]
    # archi = [(3,512), (512,512), (512,512), (512,512), (512,512), (512,512), (512,512), (512,512), (512,1)]
    # archi = [(3,512), (512,512), (512,512), (512,512), (512,1)]
    # archi = [(3,256), (256,256), (256,256), (256,256), (256,1)]
    # archi = [(3,128), (128,128), (128,128), (128,128), (128,1)]
    # archi = [(3,64), (64,64), (64,64), (64,64), (64,1)]
    # archi = [(3,64), (64,64), (64,64), (64,64), (64,64), (64,64), (64,64), (64,64), (64,1)]
    archi = [(3,32), (32,32), (32,32), (32,32), (32,32), (32,32), (32,32), (32,32), (32,1)]

    model = MultiLayerPerceptron(archi).to(config.device)

    print("PARAMETERS:", count_parameters(model))

    trainer = Trainer(dataset, config)
    trainer.add_callbacks(
        LoggerCB(os.path.join(config.output_folder, "log.txt")),
        CheckpointCB(),
        ComputeSingularValuesCB(),
    )
    trainer.train(model, full_info=True)
