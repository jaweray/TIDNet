# TIDNet Pytorch 
## Enviroment
The model is built in PyTorch 1.10.1 and tested on Ubuntu 18.04 environment, using following command to build enviroment:
> pip install -r requirement.txt

## Datasets
- `test`: Contains 112 images. Download [here](https://github.com/jaweray/TIDNet/releases/download/data/pretrained_model.pth)
- `train`: Contains 3627 images, The trainning data will be uploaded after the paper is accepted.

## Test
We provide a [pre-trained model](https://github.com/jaweray/TIDNet/releases/download/data/test.zip) for testing.

### Demoire
You can use the following command to test：
> python test.py --input_dir TEST_DATA_DIR --result_dir SAVE_PATH --weights PRETRAINED_MODEL --gpus DEVICES_ID

For example:
> python test.py --input_dir Datasets/test --result_dir results --weights pretrained_models/pretrained_model.pth --gpus 0

### OCR
Install [paddle ocr](https://www.paddlepaddle.org.cn/), then open gt_txt_dirs and modify img_ dir and out_dir, and execute it:
> python pp_ocr.py
we use the default model of paddle OCR for character recognitionf

### Evaluate
Open evaluate_end2end.py and modify img_ dir pred_txt_dirs and gt_txt_dirs, and execute it:
> python evaluate_end2end.py

## Train
Since our article is being reviewed, it is not suitable to publish the training code now. If the paper is accepted, we promise to upload the training code at the first time.
