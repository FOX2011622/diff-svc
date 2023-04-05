set PYTHONPATH=.
set CUDA_VISIBLE_DEVICES=0 
python preprocessing/binarize.py --config training/config_nsf.yaml
pause