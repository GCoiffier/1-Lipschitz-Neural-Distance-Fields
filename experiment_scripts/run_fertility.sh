# python3 experiment_scripts/prepare_fertility.py

### Lidar
python3 extract_dataset_pointcloud.py /home/coiffier/mesh/neuralSDF_inputs/fertility_scanned.obj -no 100000 -ti 5. -to -4 -visu
python3 train_lip.py fertility_lidar -model sll -n-layers 10 -n-hidden 128 -ne 300 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_lidar/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_lidar

### Good quality
python3 extract_dataset_pointcloud.py inputs/fertility_good.geogram_ascii -no 30000 -ti 3 -to 0 -visu
python3 train_lip.py fertility_good -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_good/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_good

### Sparse
python3 extract_dataset_pointcloud.py inputs/fertility_sparse.geogram_ascii -no 30000 -ti 3 -to 0 -visu
python3 train_lip.py fertility_sparse -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_sparse/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_sparse

### Noisy
python3 extract_dataset_pointcloud.py inputs/fertility_noisy.geogram_ascii -no 30000 -ti 5 -to -3 -visu
python3 train_lip.py fertility_noisy -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_noisy/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_noisy

### Ablated
python3 extract_dataset_pointcloud.py inputs/fertility_ablated.geogram_ascii -no 30000 -ti 5 -to 0. -visu
python3 train_lip.py fertility_ablated -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_ablated/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_ablated
