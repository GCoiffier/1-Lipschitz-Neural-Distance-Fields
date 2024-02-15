python extract_dataset_surface.py /home/guillaume/Desktop/neuralSDF_inputs/hand_laurent.obj -mode signed -no 300000 -ni 100000 -nt 30000

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 200 -wa 10. -lm 1e-1 -o hand_laurent_lip_sll1
python reconstruct_surface.py output/hand_laurent_lip_sll1/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll1
mv output/*.obj output/hand_laurent_lip_sll1/

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 200 -wa 10. -lm 1e-2 -o hand_laurent_lip_sll2
python reconstruct_surface.py output/hand_laurent_lip_sll2/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll2
mv output/*.obj output/hand_laurent_lip_sll2/

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 200 -wa 10. -lm 1e-3 -o hand_laurent_lip_sll3
python reconstruct_surface.py output/hand_laurent_lip_sll3/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll3
mv output/*.obj output/hand_laurent_lip_sll3/

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 200 -wa 10. -lm 1e-4 -o hand_laurent_lip_sll4
python reconstruct_surface.py output/hand_laurent_lip_sll4/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll4
mv output/*.obj output/hand_laurent_lip_sll4/

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 200 -wa 10. -lm 1e-5 -o hand_laurent_lip_sll5
python reconstruct_surface.py output/hand_laurent_lip_sll5/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll5
mv output/*.obj output/hand_laurent_lip_sll5/

python train_lip.py hand_laurent -ne 201 -cp 50 --model sll --n-layers 8 --n-hidden 32 -bs 200 -wa 10. -lm 1e-6 -o hand_laurent_lip_sll6
python reconstruct_surface.py output/hand_laurent_lip_sll6/model_final.pt -res 200 -d -0.02 -0.01 0. 0.01 0.02 -o hand_laurent_lip_sll6
mv output/*.obj output/hand_laurent_lip_sll6/
