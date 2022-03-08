# TIDNet Pytorch
pytorch+cuda11.2

## Datasets
- `test`: Contains 112 images. Download [here]()
- `train`: Contains 3627 images, The trainning data will be uploaded after the paper is accepted.

### Test
You can use the following command to test：
> python test.py --input_dir TEST_DATA_DIR --result_dir SAVE_PATH --weights PRETRAINED_MODEL --gpus DEVICES_ID

For example:
> python test.py --input_dir Datasets/test --result_dir results --weights pretrained_models/pretrained_model.pth --gpus 0

We provide a [pre-trained model]() for testing.

### Train
Since our article is being reviewed, it is not suitable to publish the training code now. If the paper is accepted, we promise to upload the training code at the first time.
