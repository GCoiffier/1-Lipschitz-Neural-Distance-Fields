python3 experiment_scripts/prepare_fertility.py

### Good quality
python3 extract_dataset_pointcloud.py inputs/fertility_good.geogram_ascii -no 10000 -ti 0.5 -to 0.5
python3 train_lip.py fertility_good -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_good/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_good

### Sparse
python3 extract_dataset_pointcloud.py inputs/fertility_sparse.geogram_ascii -no 10000 -ti 3 -to 1
python3 train_lip.py fertility_sparse -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_sparse/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_sparse

### Noisy
python3 extract_dataset_pointcloud.py inputs/fertility_noisy.geogram_ascii -no 10000 -ti 5 -to -0.1 -visu
python3 train_lip.py fertility_noisy -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_noisy/model_final.pt -res 300 -iso 0.
mv output/*.obj output/fertility_noisy

### Ablated
python3 extract_dataset_pointcloud.py inputs/fertility_ablated.geogram_ascii -no 10000 -ti 5 -to -0.1 -visu
python3 train_lip.py fertility_ablated -model sll -n-layers 10 -n-hidden 128 -ne 500 -cp 100 -bs 100
python3 reconstruct_surface.py output/fertility_ablated/model_final.pt -res 3000 -iso 0.
mv output/*.obj output/fertility_ablated