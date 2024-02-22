python3 extract_dataset_2dpolyline.py /home/guillaume/Desktop/neuralSDF_inputs/polyline_intersect.mesh -mode unsigned -no 5000 -nt 5000 -visu

python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 100 --model sll --n-layers 12 --n-hidden 128 -bs 100 -lm 1e-1 -o polyline_1

python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 100 --model sll --n-layers 12 --n-hidden 128 -bs 100 -lm 1e-2 -o polyline_2

python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 100 --model sll --n-layers 12 --n-hidden 128 -bs 100 -lm 1e-3 -o polyline_3

python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 100 --model sll --n-layers 12 --n-hidden 128 -bs 100 -lm 1e-4 -o polyline_4

python3 train_lip.py polyline_intersect --unsigned -ne 1000 -cp 100 --model sll --n-layers 12 --n-hidden 128 -bs 100 -lm 1e-5 -o polyline_5
