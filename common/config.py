from dataclasses import dataclass

@dataclass
class Config:
    dim : int = -1
    signed : bool = True
    device : str = ""
    
    batch_size : int = 100
    test_batch_size : int= 5000
    n_epochs : int = 0
    checkpoint_freq : int = 10
    
    loss_margin : float = 1e-2 # m in HKR loss
    loss_regul : float  = 100. # lambda in HKR loss
    attach_weight : float = 0.
    normal_weight : float = 0.    
    
    optimizer : str = "adam"
    learning_rate : float = 5e-4
    
    output_folder : str = ""