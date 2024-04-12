from common.models import load_model
import torch
import sys
import numpy as np
import mouette as M
from igl import signed_distance

import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use("pdf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'font.size' : 22
})

MLP = [
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_mlp/00_model_final_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_mlp/model_final.pt",
    "MLP",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_mlp_eik/20_model_final_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_mlp_eik/model_final.pt",
    r"MLP with $\mathcal{L}_{eik}$"
]

SIREN = [
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_siren/02_handle_siren_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_siren/model_final.pt",
    "SIREN",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_siren_eik/02_handle_siren_eik_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_siren_eik/model_final.pt",
    r"SIREN with $\mathcal{L}_{eik}$"
]

SAL = [
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_sal2/02_handle_sal2_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_sal2/model_final.pt",
    "SAL",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_sald2/00_model_final_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_sald2/model_final.pt",
    "SALD"
]

SLL = [
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_sll/00_model_final_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_sll/model_final.pt",
    r"SLL $\min \mathcal{L}_{fit}$",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_lip_sll/00_model_final_0.obj",
    "/home/coiffier/Bureau/SDF_results/0065_handle/handle_lip_sll/model_final.pt",
    r"SLL $\min \mathcal{L}_{hKR}$"
]


if __name__ == "__main__":

    ### Generate test dataset
    print("Generate test point cloud")
    RADIUS_MAX = 10
    N_PTS = 100_000
    P = M.sampling.sample_sphere(M.Vec.zeros(3), 1., N_PTS)
    R = np.random.exponential(3., N_PTS).reshape((N_PTS,1))
    P = R * P
    inputs = torch.Tensor(P)
    # M.mesh.save(M.mesh.from_arrays(P), "test_points.mesh")

    for meshname1,modelname1, label1, meshname2, modelname2,label2 in [
        SAL, MLP, SLL, SIREN
    ]:
        mesh1 = M.mesh.load(meshname1)
        mesh2 = M.mesh.load(meshname2)

        model1 = load_model(modelname1, "cpu")
        model2 = load_model(modelname2, "cpu")
 
        GT1,_,_ = signed_distance(P, np.array(mesh1.vertices), np.array(mesh1.faces, dtype=np.int32))
        GT1 = np.abs(GT1)        
        Y1 = np.abs(np.squeeze(model1(inputs).detach().numpy()))

        GT2,_,_ = signed_distance(P, np.array(mesh2.vertices), np.array(mesh2.faces, dtype=np.int32))
        GT2 = np.abs(GT2)
        Y2 = np.abs(np.squeeze(model2(inputs).detach().numpy()))

        plt.gca().clear()
        ax = plt.gca()
        ax.set_xlabel("Ground Truth Distance")
        ax.set_ylabel("Predicted - Real")
        ax.set_xscale('log')
        ax.set_xlim(1e-3, 2*RADIUS_MAX)
        ax.set_ylim(-0.3, 0.3)
        ax.plot([1e-3, 2*RADIUS_MAX],[0., 0.], color="black")
        ax.scatter(Y1, Y1-GT1, label=label1, alpha=0.1, s=1.)
        # if label2==r"SLL $\min \mathcal{L}_{hKR}$":
        #     ax.scatter(Y2, Y2-GT2, label=label2, alpha=0.1, s=1., color="green")
        # else:
        #     
        ax.scatter(Y2, Y2-GT2, label=label2, alpha=0.1, s=1.)
        plt.legend()
        plt.savefig(f"{label1.split()[0]}.png", bbox_inches="tight", dpi=1200)
