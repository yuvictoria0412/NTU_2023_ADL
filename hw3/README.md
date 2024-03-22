

⚠️ Please remember to specify your GPU 🥺!
## Environment setup
```
conda create -n adl-hw3 python=3.10
conda install pytorch=2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Error
when loading models
/arrow/cpp/src/arrow/filesystem/s3fs.cc:2904:  arrow::fs::FinalizeS3 was not called even though S3 was initialized.  This could lead to a segmentation fault at exit
Segmentation fault (core dumped)

### solution
```
pip install --upgrade --force-reinstall pyarrow==11.0.0
```
