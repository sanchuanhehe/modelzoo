# Copyright (c) ModelZoo. 2025-2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, argparse, cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import multiprocessing

model_config = {
    384: {'resize': 384, 'centercrop': 384, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},  # resize 384
    224: {'resize': 248, 'centercrop': 224, 'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},  # resize 224
}

# 获取预处理后的bin图片
def get_preprocessed_bin(img_batches, batch, store_path, image_size):
    # 图像缩放
    def img_resize(img, size, interpolation=Image.BICUBIC):
        if not isinstance(size, int):
            return img.resize(size[::-1], interpolation)

        width, height = img.size
        if (width <= height and width == size) or (height <= width and height == size):
            return img

        if width < height:
            reWidth = size
            reHeight = int(size * height / width)
            return img.resize((reWidth, reHeight), interpolation)
        reHeight = size
        reWidth = int(size * width / height)
        ret_img =  img.resize((reWidth, reHeight), interpolation)
        return ret_img
    
    # 中心裁剪    
    def crop_center(img, output_size):
        output_size = (int(output_size), int(output_size))
        img_width, img_height = img.size
        crop_height, crop_width = output_size
        top = int(round((img_height - crop_height) / 2.))
        left = int(round((img_width - crop_width) / 2.))
        ret_img = img.crop((left, top, left + crop_width, top + crop_height))
        return ret_img

    # 归一化
    def normalization(pic, avgs, varis):
        pic = np.array(pic)
        for chn, (avg, vari) in enumerate(zip(avgs, varis)):
            pic[:, :, chn] = (pic[:, :, chn] - avg) / vari
        return pic
    # 遍历图像批次

    for filename, filename_path in tqdm(img_batches[batch]):
        pic = Image.open(filename_path).convert('RGB')
        # 这里可以将image 转成f32 类型的吗，能否减少下面resize 的精度损失
        pic = img_resize(pic, model_config[image_size]['resize']) # Resize
        pic = crop_center(pic, model_config[image_size]['centercrop']) # CenterCrop

        # 转换成浮点数，并归一到[0,1]之间
        pic = np.array(pic).astype(np.float32) / 255

        # 均值归一化
        pic = normalization(pic, model_config[image_size]['mean'], model_config[image_size]['std']) # Normalization
        # 将HWC 调整成 CHW 排布
        pic = pic.transpose((2, 0, 1))
        # 保存预处理后的结果 bin
        pic.tofile(os.path.join(store_path, filename.split('.')[0] + ".bin"))

def preprocess_data(data_path, store_path, image_size):
    # 文件列表
    print("preprocess_data doing...")
    file_list = os.listdir(data_path)
    img_list_info = []
    for filename in file_list:
        filename_path = os.path.join(data_path, filename)
        if os.path.isdir(filename_path):
            img_list_info += \
                [(image_name, os.path.join(filename_path, image_name)) for image_name in os.listdir(filename_path)]
        else:
            img_list_info.append((filename, filename_path))
    # 单个线程处理的批次大小，相应的有10个线程处理
    thread_batchsize = 5000
    img_list_info.sort()
    if len(img_list_info) < thread_batchsize:
        img_batches = [img_list_info[0 : len(img_list_info)]]
    else:
        #单个批次的图片数组
        img_batches = [img_list_info[i:i + thread_batchsize] for i in range(0, len(img_list_info), thread_batchsize) if img_list_info[i:i + thread_batchsize] != []]

    thrd_pool = multiprocessing.Pool(len(img_batches))
    for img_batch in range(len(img_batches)):
        thrd_pool.apply_async(get_preprocessed_bin, args=(img_batches, img_batch, store_path, image_size))
    # 关闭线程池
    thrd_pool.close()
    thrd_pool.join()
    print("process done.")


if __name__ == "__main__":
    # 解析参数
    parser_args = argparse.ArgumentParser(description='Path', add_help=False)
    # 预处理后的路径
    parser_args.add_argument('--store_path', default=r"data/img", metavar='DIR',  help='Save the preprocessed results')
    # 源数据路径
    parser_args.add_argument('--data_path', default=r"../../../../datasets/ImageNet/val", metavar='DIR',
        help='Original dataset')
    # 预处理后的大小
    parser_args.add_argument('--image_size', type=int, required=True, help='Size of the preprocessed image.')

    args = parser_args.parse_args()
    # 创建文件夹
    if not os.path.exists(args.store_path):
        os.makedirs(args.store_path)
    # 预处理
    preprocess_data(args.data_path, args.store_path, args.image_size)
