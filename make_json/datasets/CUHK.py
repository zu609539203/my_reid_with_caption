# -*- coding:utf-8 _*-
"""
@Author: Bing Tang
@Time: “2022/6/28 10:04”
"""
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os

from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299#最大维度，将patch的图片的维度调整

def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image    #返回resize过大小的图片

# img_dir = '../test_img/p1_s3.jpg'
# img = Image.open(img_dir)
# new_img = under_max(img)
# print(new_img.size)
# new_img.show()

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)

#数据增强
train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # tv.transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
    #         std = [ 0.229, 0.224, 0.225 ])
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class CUHK_PEDES(Dataset):
    def __init__(self, conf, data_info, max_length, transform = train_transform, mode = 'training'):
        # self.split = data_info[0]["split"]
        #self.is_train = is_train
        self.conf = conf
        self.data_info = data_info
        self.annot = [(data["id"],data["captions"][random.randint(0,1)])
                     for data in data_info]
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower = True)
        self.max_length = max_length + 1

    def __getitem__(self, idex):
        image_path = os.path.join(self.conf.CUHKdata, self.data_info[idex]["file_path"])
        img = Image.open(image_path)
        img = self.transform(img)
        img = nested_tensor_from_tensor_list(img.unsqueeze(0))
        cap_index = random.randint(0,1)
        id, caption = self.annot[idex]
        # caption = self.data_info[idex]["captions"][cap_index]
        caption_encoded = self.tokenizer.encode_plus(caption, max_length=self.max_length,
        pad_to_max_length = True, return_attention_mask=True, return_token_type_ids=False)

        caption_encoded = self.tokenizer.encode_plus(caption, max_length=self.max_length, pad_to_max_length=True,
        return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return img.tensors.squeeze(0), img.mask.squeeze(0), caption, cap_mask

    def __len__(self):
        return len(self.annot)

def build_dataset(configure, mode = "training"):
    if mode == 'training':
        train_dir = configure.CUHKano
        train_file = os.path.join(train_dir, 'train_set.json')
        data = CUHK_PEDES(configure, data_info = read_json(train_file), max_length= configure.max_position_embeddings,
                          transform = train_transform, mode = 'training')
        return data

    elif mode == 'validation':
        val_dir = configure.CUHKano
        val_file = os.path.join(val_dir, 'valid_set.json')
        data = CUHK_PEDES(configure, data_info = read_json(val_file), max_length = configure.max_position_embeddings,
                          transform = val_transform, mode = 'validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")