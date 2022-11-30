# TIDNet Pytorch 
## Enviroment
The model is built in PyTorch 1.10.1 and tested on Ubuntu 18.04 environment, using following command to build enviroment:
> pip install -r requirement.txt

## Datasets
- `test`: Contains 112 images. Download [here](https://github.com/jaweray/TIDNet/releases/download/data/test.zip)
- `train`: Contains 3627 images. The trainning data will be uploaded after the paper is accepted.

## Test
We provide a [pre-trained model](https://github.com/jaweray/TIDNet/releases/download/data/pretrained_model.pth) for testing.

### Demoire
You can use the following command to test：
> python test.py --input_dir TEST_DATA_DIR --result_dir SAVE_PATH --weights PRETRAINED_MODEL --gpus DEVICES_ID

For example:
> python test.py --input_dir Datasets/test --result_dir results --weights pretrained_models/pretrained_model.pth --gpus 

### OCR
Install [paddleOCR](https://www.paddlepaddle.org.cn/), then open paddle_ocr.py and modify its **img_ dir** and **out_dir** to demoire images directory and OCR prediction output directory, and then execute:
> python paddle_ocr.py
We use the PP-OCRv2 model of paddleOCR for character recognition.

### Evaluate
Open evaluate_end2end.py and modify **pred_txt_dirs** and **gt_txt_dirs** to OCR prediction output directory and ground-truth directory, and then execute it:
> python evaluate_end2end.py

## Train
Since our article is being reviewed, it is not suitable to publish the training code now. If the paper is accepted, we promise to upload the training code at the first time.
