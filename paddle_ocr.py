import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from paddleocr import PaddleOCR, draw_ocr
import json
import numpy as np
import cv2

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def is_json(filename):
    return any(filename.endswith(extension) for extension in ['json'])


def polygon_crop(img, points):
    list = [[int(x) for x in lst] for lst in points]

    list[0][0] -= 10
    list[0][1] -= 10

    list[1][0] += 10
    list[1][1] -= 10

    list[2][0] += 10
    list[2][1] += 10

    list[3][0] -= 10
    list[3][1] += 10

    pts = np.array(list)

    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y:y + h, x:x + w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst


img_ext = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
img_dir = r'C:\Users\Administrator\Desktop\tt1\tidnet\demoire_images'
out_dir = r'C:\Users\Administrator\Desktop\tt1\tidnet\ocr_prediction'
inp_files = sorted(os.listdir(img_dir))
inp_filenames = [os.path.join(img_dir, x) for x in inp_files if os.path.splitext(x)[1] in img_ext]
txt_filenames = [os.path.join(out_dir, os.path.splitext(x)[0] + '.txt') for x in inp_files if os.path.splitext(x)[1] in img_ext]
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(cls_model_dir='./paddle_ocr_model/ch_ppocr_mobile_v2.0_cls_infer',
                det_model_dir='./paddle_ocr_model/ch_PP-OCRv2_det_infer',
                rec_model_dir='./paddle_ocr_model/ch_PP-OCRv2_rec_infer')  # need to run only once to download and load model into memory

recall = 0
accuracy = 0
sizex = len(inp_filenames)  # get the size of target
for index in range(sizex):
    img_path = inp_filenames[index]
    txt_path = txt_filenames[index]

    result = ocr.ocr(img_path)
    with open(txt_path, 'w', encoding='utf8') as txt_fp:
        for line in result:
            points = line[0]
            label = line[1][0]
            points = [str(i) for j in points for i in j]
            txt_fp.write(','.join(points) + '\t' + label + '\n')
    # print(img_path)
    # print(tar_path)
    #
    # with open(tar_path, 'r', encoding='utf8') as fp:
    #     json_data = json.load(fp)['shapes']
    #     # result = ocr.ocr(img_path, cls=True)
    #     # result_size = len(result)
    #     target_size = len(json_data)
    #     sum = 0
    #     img = cv2.imread(img_path)
    #     i = 0
    #     for target in json_data:
    #         i += 1
    #         box = target['points']
    #         temp_img = polygon_crop(img, box)
    #         cv2.imwrite(os.path.join(temp_dir, str(i) + 'temp.png'), temp_img)
    #         result = ocr.ocr(os.path.join(temp_dir, str(i) + 'temp.png'), cls=False)
    #
    #         if result:
    #             print(result)
    #             print(i, '|'+result[0][1][0]+'|', '|'+target['label']+'|', result[0][1][0] == target['label'])
    #             if result[0][1][0] == target['label']:
    #                 sum += 1
            # for output in result:
            #     if target['label'] == output[1][0]:
            #         sum += 1
            #         break
#         if result_size != 0:
#             recall += sum / result_size
#         accuracy += sum / target_size
#         print('{:.4f}\t{:.4f}'.format(recall, accuracy))
#         accuracy += sum / target_size
#     print('Accuracy: {:.4f}'.format(accuracy))
# recall /= sizex
# accuracy /= sizex
# print('Recall: {:.4f}\tAccuracy: {:.4f}'.format(recall, accuracy))

# # 显示结果
# from PIL import Image
#
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')