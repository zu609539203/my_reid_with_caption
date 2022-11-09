"""
note:
    将 big_train 内的照片[包含已有注释和无注释]进行重新打包，并且将没有注释的生成注释
格式：
    {
        'file_name' : 'xxx.jpg',
        'captions' : 'The man xxx xx xxx'
    }
"""

import os
import torch
import argparse
import json
import random
import glob
from tqdm import tqdm

from transformers import BertTokenizer
from PIL import Image

from models import caption
from datasets import CUHK
from configuration import Config

parser = argparse.ArgumentParser(description='Image Captioning')

img_path = '/home/lkh/zch/person_reid_language_v1-master/data/zch/CUHK-SYSU/cropped_images/query_person/*.jpg'
cap_path = '/home/lkh/zch/person_reid_language_v1-master/data/zch/CUHK-PEDES/caption_SYSU.json'

with open(cap_path) as fin:
    data = json.load(fin)
all_captions = {}
for cap in data:
    all_captions[cap['file_path']] = cap['captions']

config = Config()
device = torch.device(config.device)
checkpoint_path = 'checkpoints/checkpoint_630new.pth'

model, _ = caption.build_model(config)

print("Loading Checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location=config.device)
model.load_state_dict(checkpoint['model'])
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


@torch.no_grad()
def evaluate():
    # model.to(device)
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image.to(device), caption.to(device), cap_mask.to(device))
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i + 1] = predicted_id[0]
        cap_mask[:, i + 1] = False

    return caption

# 存储 CUHK-PEDES 的路径
cusy_path = '/home/lkh/zch/person_reid_language_v1-master/data/zch/CUHK-PEDES/imgs/CUHK-SYSU'

list = []
# i = 0
for img in tqdm(glob.glob(img_path), ncols= 80):
    # i+=1
    # if(i == 200):
    #     break
    imgs = img.split('/')
    img_name = imgs[-1]

    result = ''
    image_path = os.path.join(cusy_path, img_name)

    if os.path.exists(image_path):
        key_image_list = glob.glob(image_path)
        key_image = random.choice(key_image_list).replace(cusy_path, 'CUHK-SYSU')

        sentence = all_captions[key_image]
        num_captions = random.randint(0, 1)
        single_sentence = sentence[num_captions]

    if not os.path.exists(image_path):
        image = Image.open(img)

        image = CUHK.val_transform(image)
        image = image.unsqueeze(0)
        caption, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

        output = evaluate()
        single_sentence = tokenizer.decode(output[0].tolist(), skip_special_tokens=True).capitalize()

    filename = "query_person_caption.json"
    dict = {
        'file_path': img_name,
        'captions': single_sentence
    }
    list.append(dict)

with open(filename, 'w') as obj:
    json.dump(list, obj, indent=2)
















