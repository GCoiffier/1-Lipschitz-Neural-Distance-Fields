for fullfile in $1*
do
    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    filename="${filename%.*}"
    echo $filename
    python3 extract_dataset_surface.py $fullfile -no 50000 -ni 10000 -nt 10000
    python3 train_lip.py $filename -ne 1000 -cp 100 --model sll --n-layers 10 --n-hidden 256 -bs 100
done