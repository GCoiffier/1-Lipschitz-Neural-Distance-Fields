import matplotlib.pyplot as plt
from common.models import load_model
import torch
import sys
import numpy as np
import mouette as M
from igl import signed_distance

if __name__ == "__main__":

    mesh_path = sys.argv[1]
    models = sys.argv[2:]

    mesh = M.mesh.load(mesh_path)

    nmodels = len(models)
    print(f"Provided {nmodels} models")
    assert nmodels>0

    ### Generate test dataset
    N_RAYS = 20
    N_RADII = 50
    RMAX = 10
    P = np.vstack([
        np.random.normal(0.,1., size=N_RAYS),
        np.random.normal(0.,1., size=N_RAYS),
        np.random.normal(0.,1., size=N_RAYS)
    ]).T    
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    P = np.concatenate([r*P for r in np.logspace(-1, np.log10(RMAX),N_RADII)])
    GT,_,_ = signed_distance(P, np.array(mesh.vertices), np.array(mesh.faces, dtype=np.int32))
    inputs = torch.Tensor(P)
    
    maxi = 2*np.amax(GT)
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel("Ground Truth Distance")
    ax.set_ylabel("Predicted Distance")
    ax.set_xlim(1e-3, maxi)
    ax.set_ylim(1e-3, maxi)
    ax.plot([1e-3, maxi],[1e-3, maxi], color="black", alpha=0.5)

    for model_path in models:
        model_name = M.utils.get_filename(model_path)
        try:
            model = load_model(model_path, "cpu")
            Y = model(inputs).detach().numpy()
            ax.scatter(GT,Y,label=model_name, alpha=0.3, s=10.)
        except Exception as e:
            print(f"ERROR with model {model_name} : {e}")
            continue
    plt.legend()
    plt.show()