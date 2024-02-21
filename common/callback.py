from .models import save_model
from .visualize import render_sdf_2d, parameter_singular_values, reconstruct_surface_marching_cubes
import mouette as M
import os
import csv

class Callback:
    """
    An empty Callback object to be called inside a Trainer (see trainer.py)

    Callback affect the trainer they are associated with, or provide log infos, or anything you can think of.
    Inside a Trainer, they can be called at three points:
    - At the beginning of an training epoch
    - At the end of an training epoch
    - At the end of a forward/backward pass
    - At the end of a testing epoch
    - At the very end of the training procedure
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def callOnEndForward(self, trainer, model):
        pass

    def callOnBeginTrain(self, trainer, model):
        pass
    
    def callOnEndTrain(self, trainer, model):
        pass

    def callOnEndTest(self, trainer, model):
        pass
    

class LoggerCB(Callback):
    """
    Registers metrics of the training inside a .log file
    """
    
    def __init__(self, file_path):
        super().__init__()
        self.path = file_path
        self.logged = {"epoch" : -1, "time" : 0, "train_loss" : -1, "test_loss" : -1}
        with open(self.path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writeheader()

    def callOnEndTrain(self, trainer, model):
        self.logged.update({"epoch" :  trainer.metrics["epoch"]})
        self.logged.update({"time" :  trainer.metrics["epoch_time"]})
        self.logged.update({"train_loss" : trainer.metrics["train_loss"]})
        trainer.log(f"Train loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['train_loss']}")

    def callOnEndTest(self, trainer, model):
        self.logged.update({"test_loss" : trainer.metrics["test_loss"]})
        trainer.log(f"Test loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['test_loss']}")
        print()
        self._write_log()

    def _write_log(self):
        with open(self.path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writerow(self.logged)

class CheckpointCB(Callback):
    """
    A Specific Callback responsible for saving the model currently in training into a file
    """
    def __init__(self, when : dict):
        self.when = when

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            name = f"model_e{epoch}.pt"
            path = os.path.join(trainer.config.output_folder, name)
            save_model(model, path)


class ComputeSingularValuesCB(Callback):

    def __init__(self, freq):
        self.freq = freq

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            singular_values = parameter_singular_values(model)
            print("Singular values:")
            for sv in singular_values:
                print(sv)
            print()


class Render2DCB(Callback):

    def __init__(self, save_folder, freq, plot_domain, res=800, output_contours=True, output_gradient_norm=True):
        super().__init__()
        self.save_folder = save_folder
        self.freq = freq
        self.domain = plot_domain
        self.freq = freq
        self.res = res
        self.output_contours = output_contours
        self.output_gradient_norm = output_gradient_norm

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            render_path = os.path.join(self.save_folder, f"render_{epoch}.png")
            contour_path = os.path.join(self.save_folder, f"contour_{epoch}.png") if self.output_contours else None
            gradient_path = os.path.join(self.save_folder, f"grad_{epoch}.png") if self.output_gradient_norm else None
            render_sdf_2d(
                render_path,
                contour_path,
                gradient_path,
                model, 
                self.domain, 
                trainer.config.device, 
                res=self.res, 
                batch_size=trainer.config.test_batch_size,
            )

class MarchingCubeCB(Callback):
    def __init__(self, save_folder, freq, domain, res=100, iso=0):
        super().__init__()
        self.save_folder = save_folder
        self.freq = freq
        self.domain = domain
        self.res = res
        if isinstance(iso,float):
            self.iso = [iso]
        else:
            self.iso = iso
    
    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if self.freq>0 and epoch%self.freq==0:
            iso_surfaces = reconstruct_surface_marching_cubes(
                model, 
                self.domain, 
                trainer.config.device, 
                self.iso, 
                self.res, 
                trainer.config.test_batch_size)
            for (n,off),mesh in iso_surfaces.items():
                M.mesh.save(mesh, os.path.join(self.save_folder, f"e{epoch:04d}_n{n:02d}_iso{round(1000*off)}.obj"))
    
class UpdateHkrRegulCB(Callback):

    def __init__(self, when : dict):
        super().__init__()
        self.when = when

    def callOnBeginTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            trainer.config.loss_regul = self.when[epoch]
            trainer.log("Updated loss regul weight to", self.when[epoch])