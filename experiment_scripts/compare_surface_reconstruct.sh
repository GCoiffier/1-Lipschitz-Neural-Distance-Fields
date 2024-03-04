# Run ground truth versions
python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode dist -no 100000 -ni 50000 -nt 10000

python train_fullinfo.py hand_laurent -ne 500 -cp 100 --model mlp --n-layers 20 --n-hidden 128 -bs 100 -o hand_laurent_mlp
python reconstruct_surface.py output/hand_laurent_mlp/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_mlp
mv output/*.obj output/hand_laurent_mlp/

python train_fullinfo.py hand_laurent -ne 500 -cp 100 --model mlp --n-layers 20 --n-hidden 128 -bs 100 -weik 0.1 -o hand_laurent_mlp_eik
python reconstruct_surface.py output/hand_laurent_mlp_eik/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_mlp_eik
mv output/*.obj output/hand_laurent_mlp_eik/

python train_fullinfo.py hand_laurent -ne 500 -cp 100 --model siren --n-layers 4 --n-hidden 256 -bs 1000 -lr 1e-4 -o hand_laurent_siren
python reconstruct_surface.py output/hand_laurent_siren/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_siren
mv output/*.obj output/hand_laurent_siren/

python train_fullinfo.py hand_laurent -ne 500 -cp 100 --model siren --n-layers 4 --n-hidden 256 -bs 1000 -lr 1e-4 -weik 0.1 -o hand_laurent_siren_eik
python reconstruct_surface.py output/hand_laurent_siren_eik/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_siren_eik
mv output/*.obj output/hand_laurent_siren_eik/

# Run signed lipschitz with full info
python train_fullinfo.py hand_laurent -ne 500 -cp 100 --model sll --n-layers 20 --n-hidden 128 -bs 100 -o hand_laurent_sll
python reconstruct_surface.py output/hand_laurent_lip_sll/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_sll
mv output/*.obj output/hand_laurent_sll/

# Run SALD
python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode sal -no 100000 -ni 50000 -nt 100

python train_SALD.py hand_laurent -ne 500 -cp 100 -model mlp --n-layers 20 --n-hidden 128 -bs 100 -wa 1. -o hand_laurent_sal2
python reconstruct_surface.py output/hand_laurent_lip_sal2/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_sal2
mv output/*.obj output/hand_laurent_lip_sal2/

python train_SALD.py hand_laurent -ne 500 -cp 100 -model mlp --n-layers 20 --n-hidden 128 -bs 100 -wa 1. --metric l0 -o hand_laurent_sal0
python reconstruct_surface.py output/hand_laurent_lip_sal0/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_sal0
mv output/*.obj output/hand_laurent_lip_sal0/

# Run signed lipschitz with HKR
python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode signed -no 100000 -ni 50000 -nt 10000

python train_lip.py hand_laurent -ne 500 -cp 50 --model sll --n-layers 20 --n-hidden 128 -bs 1000 -lm 1e-3 -wa 1. -o hand_laurent_lip_sll
python reconstruct_surface.py output/hand_laurent_lip_sll/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll
mv output/*.obj output/hand_laurent_lip_sll/

python train_lip.py hand_laurent -ne 300 -cp 50 --model ortho --n-layers 20 --n-hidden 128 -bs 1000 -lm 1e-3 -wa 1. -o hand_laurent_lip_ortho
python reconstruct_surface.py output/hand_laurent_lip_ortho/model_final.pt -res 300 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_ortho
mv output/*.obj output/hand_laurent_lip_ortho/

# Run unsigned lipschitz
# python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode unsigned -no 100000 -ni 50000 -nt 10000

# python train_lip.py hand_laurent --unsigned -ne 101 -cp 50 --model ortho --n-layers 20 --n-hidden 128 -bs 200 -wa 10. -o hand_laurent_lip_unsigned_ortho
# python reconstruct_surface.py output/hand_laurent_lip_unsigned_ortho/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_unsigned_ortho
# mv output/*.obj output/hand_laurent_lip_unsigned_ortho/

# python train_lip.py hand_laurent --unsigned -ne 201 -cp 50 --model sll --n-layers 20 --n-hidden 128 -bs 200 -wa 10. -o hand_laurent_lip_unsigned_sll
# python reconstruct_surface.py output/hand_laurent_lip_unsigned_sll/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_unsigned_sll
# mv output/*.obj output/hand_laurent_lip_unsigned_sll/
