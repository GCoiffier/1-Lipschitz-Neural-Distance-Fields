# Run ground truth versions
python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode dist --importance-sampling -no 500000 -ni 100000 -nt 20000

python train_fullinfo.py hand_laurent -ne 201 -cp 50 --model mlp   --n-layers 8 --n-hidden 32 -bs 100 -o hand_laurent_mlp
python reconstruct_surface.py output/hand_laurent_mlp/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_mlp
mv output/*.obj output/hand_laurent_mlp/

python train_fullinfo.py hand_laurent -ne 201 -cp 50 --model mlp   --n-layers 8 --n-hidden 32 -bs 100 -weik 0.1 -o hand_laurent_mlp_eik
python reconstruct_surface.py output/hand_laurent_mlp_eik/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_mlp_eik
mv output/*.obj output/hand_laurent_mlp_eik/

python train_fullinfo.py hand_laurent -ne 201 -cp 50 --model siren --n-layers 8 --n-hidden 32 -bs 100 -o hand_laurent_siren
python reconstruct_surface.py output/hand_laurent_siren/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_siren
mv output/*.obj output/hand_laurent_siren/

python train_fullinfo.py hand_laurent -ne 201 -cp 50 --model siren --n-layers 8 --n-hidden 32 -bs 100 -weik 0.1 -o hand_laurent_siren_eik
python reconstruct_surface.py output/hand_laurent_siren_eik/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_siren_eik
mv output/*.obj output/hand_laurent_siren_eik/

# Run signed lipschitz 
python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode signed -no 500000 -ni 100000 -nt 20000

python train_lip.py hand_laurent -ne 201 -cp 50 --model ortho --n-layers 8 --n-hidden 32 -bs 100 -wa 10. -o hand_laurent_lip_ortho
python reconstruct_surface.py output/hand_laurent_lip_ortho/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_ortho
mv output/*.obj output/hand_laurent_lip_ortho/

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 100 -wa 10. -o hand_laurent_lip_sll
python reconstruct_surface.py output/hand_laurent_lip_sll/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll
mv output/*.obj output/hand_laurent_lip_sll/

# Run unsigned lipschitz
python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode unsigned -no 500000 -ni 100000 -nt 20000

python train_lip.py hand_laurent --unsigned -ne 201 -cp 50 --model ortho --n-layers 8 --n-hidden 32 -bs 100 -wa 10. -o hand_laurent_lip_unsigned_ortho
python reconstruct_surface.py output/hand_laurent_lip_unsigned_ortho/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_unsigned_ortho
mv output/*.obj output/hand_laurent_lip_unsigned_ortho/

python train_lip.py hand_laurent --unsigned -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 100 -wa 10. -o hand_laurent_lip_unsigned_sll
python reconstruct_surface.py output/hand_laurent_lip_unsigned_sll/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_unsigned_sll
mv output/*.obj output/hand_laurent_lip_unsigned_sll/