python extract_dataset_pointcloud.py ~/Desktop/neuralSDF_inputs/vase_point_cloud.geogram_ascii -no 100000 -ti 4 -to -3 -visu
python train_lip.py vase_point_cloud -model sll --n-layers 10 --n-hidden 128 -ne 100 -cp 20 -bs 100 -o botijo_signed
python reconstruct_surface.py output/botijo_signed/model_final.pt -res 300 -iso 0.
mv output/*.obj output/botijo_signed
mv inputs/*.geogram_ascii output/botijo_signed/

python extract_dataset_pointcloud.py ~/Desktop/neuralSDF_inputs/vase_point_cloud.geogram_ascii -mode unsigned -visu
python train_lip.py vase_point_cloud --unsigned -model sll --n-layers 10 --n-hidden 128 -ne 200 -cp 40 -bs 100 -lm 1e-2 -o botijo_unsigned
python reconstruct_surface.py output/botijo_unsigned/model_final.pt -res 300 -iso 0.
mv output/*.obj output/botijo_unsigned
mv inputs/*.geogram_ascii output/botijo_unsigned/
