from deel import torchlip

def DenseLipNetwork(
    dim_in:float,
    dim_hidden:float,
    n_layers:int,
    group_sort_size:int=0, 
    k_coeff_lip:float=1., 
    niter_spectral:int=3,
    niter_bjorck:int=15
):
    layers = []
    activation = torchlip.FullSort if group_sort_size == 0 else lambda : torchlip.GroupSort(group_sort_size)
    layers.append(torchlip.SpectralLinear(dim_in, dim_hidden, niter_spectral=niter_spectral, niter_bjorck=niter_bjorck))
    layers.append(activation())
    for _ in range(n_layers-1):
        layers.append(torchlip.SpectralLinear(dim_hidden, dim_hidden, niter_spectral=niter_spectral, niter_bjorck=niter_bjorck))
        layers.append(activation())
    layers.append(torchlip.FrobeniusLinear(dim_hidden, 1))
    model = torchlip.Sequential(*layers, k_coef_lip=k_coeff_lip)
    model.meta = [dim_in, dim_hidden, n_layers]
    model.id = "Spectral"
    return model