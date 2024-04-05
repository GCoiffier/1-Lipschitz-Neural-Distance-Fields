# Run ground truth versions
# python extract_dataset_surface.py /home/coiffier/mesh/neuralSDF_inputs/handle.STL -mode dist -no 100000 -ni 50000 -nt 10000 -visu

# python train_fullinfo.py handle -ne 100 --model mlp --n-layers 8 --n-hidden 256 -bs 100 -o handle_mlp
# python reconstruct_surface.py output/handle_mlp/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_mlp
# mv output/*.obj output/handle_mlp/

# python train_fullinfo.py handle -ne 100 --model mlp --n-layers 8 --n-hidden 256 -bs 100 -weik 0.1 -o handle_mlp_eik
# python reconstruct_surface.py output/handle_mlp_eik/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_mlp_eik
# mv output/*.obj output/handle_mlp_eik/

# python train_fullinfo.py handle -ne 100 --model siren --n-layers 8 --n-hidden 256 -bs 1000 -lr 1e-4 -o handle_siren
# python reconstruct_surface.py output/handle_siren/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_siren
# mv output/*.obj output/handle_siren/

# python train_fullinfo.py handle -ne 100 --model siren --n-layers 8 --n-hidden 256 -bs 1000 -lr 1e-4 -weik 0.1 -o handle_siren_eik
# python reconstruct_surface.py output/handle_siren_eik/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_siren_eik
# mv output/*.obj output/handle_siren_eik/

# Run signed lipschitz with full info
# python train_fullinfo.py handle -ne 100 --model sll --n-layers 8 --n-hidden 256 -bs 100 -o handle_sll
# python reconstruct_surface.py output/handle_lip_sll/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_sll
# mv output/*.obj output/handle_sll/

# Run SALD
python extract_dataset_surface.py /home/coiffier/mesh/neuralSDF_inputs/handle.STL -mode sal -no 100000 -ni 50000 -nt 100

python train_SALD.py handle -ne 200 -cp 100 -model mlp --n-layers 8 --n-hidden 256 -bs 100 -wa 1. -wg 0. -o handle_sal2
python reconstruct_surface.py output/handle_sal2/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_sal2
mv output/*.obj output/handle_sal2/

python train_SALD.py handle -ne 200 -cp 100 -model mlp --n-layers 8 --n-hidden 256 -bs 100 -wa 1. -o handle_sald2
python reconstruct_surface.py output/handle_sald2/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_sald2
mv output/*.obj output/handle_sald2/

# python train_SALD.py handle -ne 100 -model mlp --n-layers 8 --n-hidden 256 -bs 100 -wa 1. --metric l0 -o handle_sal0
# python reconstruct_surface.py output/handle_lip_sal0/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_sal0
# mv output/*.obj output/handle_sal0/

# Run signed lipschitz with HKR
# python extract_dataset_surface.py /home/coiffier/mesh/neuralSDF_inputs/handle.STL -mode signed -no 100000 -ni 50000 -nt 10000

# python train_lip.py handle -ne 100 -cp 50 --model sll --n-layers 8 --n-hidden 256 -bs 1000 -lm 1e-3 -wa 1. -o handle_lip_sll
# python reconstruct_surface.py output/handle_lip_sll/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_lip_sll
# mv output/*.obj output/handle_lip_sll/

# python train_lip.py handle -ne 300 -cp 50 --model ortho --n-layers 8 --n-hidden 128 -bs 1000 -lm 1e-3 -wa 1. -o handle_lip_ortho
# python reconstruct_surface.py output/handle_lip_ortho/model_final.pt -res 300 -iso -0.02 -0.01 0. 0.01 0.02 -o handle_lip_ortho
# mv output/*.obj output/handle_lip_ortho/
