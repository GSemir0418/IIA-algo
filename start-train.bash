# preprocess dataset
cd data/op-data
python3 extract_labels.py
# start train
cd ../..
python3 train.py
# predict
python3 predict.py