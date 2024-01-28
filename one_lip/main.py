import os
import argparse
from types import SimpleNamespace

import mouette as M

from common.dataset import PointCloudDataset
from common.model import *
from common.visualize import *
from common.training import Trainer
from common.utils import get_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("dataset", type=str)
    parser.add_argument("-o", "--output-name", type=str, default="")
    parser.add_argument("-n", "--n-iter", type=int, default=10, help="Number of iterations")
    parser.add_argument('-bs',"--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("-ne", "--epochs", type=int, default=10, help="Number of epochs per iteration")
    parser.add_argument("-cpu", action="store_true")
    parser.add_argument("-a", "--attach-weight", type=float, default=0.)
    args = parser.parse_args()

    #### Config ####
    config = SimpleNamespace(
        dim = 3,
        device = get_device(args.cpu),
        n_iter = args.n_iter,
        batch_size = args.batch_size,
        test_batch_size = 5000,
        epochs = args.epochs,
        loss_margin = 0.01, # m
        loss_regul = 100., # lambda
        loss_attach_weight = args.attach_weight,
        optimizer = "adam",
        learning_rate = 1e-4,
        update_distrib = False,
        NR_maxiter = 3,
        output_folder = os.path.join("output", args.output_name if len(args.output_name)>0 else args.dataset)
    )
    os.makedirs(config.output_folder, exist_ok=True)
    print("DEVICE:", config.device)

    #### Load dataset ####
    dataset = PointCloudDataset(args.dataset, config)
    plot_domain = dataset.object_BB
    plot_domain.pad(0.5, 0.5, 0.5)

    #### Create model and setup trainer
    archi = [(3,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1024), (1024,1)]
    # archi = [(3,512), (512,512), (512,512), (512,512), (512,512), (512,512), (512,512), (512,512), (512,1)]
    # archi = [(3,512), (512,512), (512,512), (512,512), (512,1)]
    # archi = [(3,256), (256,256), (256,256), (256,256), (256,1)]
    # archi = [(3,128), (128,128), (128,128), (128,128), (128,1)]
    # archi = [(3,64), (64,64), (64,64), (64,64), (64,1)]
    # archi = [(3,32), (32,32), (32,32), (32,32), (32,1)]

    model = DenseLipNetwork(
        archi, group_sort_size=0,
        niter_spectral=3, niter_bjorck=15
    ).to(config.device)

    print("PARAMETERS:", count_parameters(model))
    pc = point_cloud_from_tensors(
        dataset.X_train_bd.detach().cpu(), 
        dataset.X_train_in.detach().cpu(), 
        dataset.X_train_out.detach().cpu())
    M.mesh.save(pc, os.path.join(config.output_folder, "pc_0.geogram_ascii"))

    for n in range(config.n_iter):
        if n==0: config.loss_regul = 1.
        if n==1: config.loss_regul = 100.
    
        print("ITERATION", n+1)
        trainer = Trainer(dataset, config)
        trainer.train(model)
        
        singular_values = parameter_singular_values(model)
        print("Singular values:")
        for sv in singular_values:
            print(sv)
        print()
        model_path = os.path.join(config.output_folder, f"model_{n+1}.pt")
        save_model(model, archi, model_path)
        if config.update_distrib:
            dataset.update_complementary_distribution(model, config.NR_maxiter)
            pc = point_cloud_from_tensors(
                dataset.X_train_bd.detach().cpu(), 
                dataset.X_train_in.detach().cpu(), 
                dataset.X_train_out.detach().cpu())
            M.mesh.save(pc, os.path.join(config.output_folder, f"pc_{n+1}.geogram_ascii"))
