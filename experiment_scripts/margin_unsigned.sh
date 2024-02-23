python3 extract_dataset_2dpolyline.py /home/guillaume/Desktop/neuralSDF_inputs/polyline_intersect.mesh -mode unsigned -no 5000 -nt 5000 -visu

# python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 500 --model sll --n-layers 12 --n-hidden 128 -bs 1000 -lm 1e-1 -o polyline_1e-1

python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 100 --model sll --n-layers 12 --n-hidden 128 -bs 100 -lm 2e-2 -o polyline_2e-1

# python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 500 --model sll --n-layers 12 --n-hidden 128 -bs 1000 -lm 1e-2 -o polyline_1e-2

# python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 500 --model sll --n-layers 12 --n-hidden 128 -bs 1000 -lm 1e-3 -o polyline_1e-3

# python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 500 --model sll --n-layers 12 --n-hidden 128 -bs 1000 -lm 1e-4 -o polyline_1e-4
