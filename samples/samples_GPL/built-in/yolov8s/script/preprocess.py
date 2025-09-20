import os
import cv2
import numpy as np
import argparse

def letterbox(img, target_size=(640, 640)):
    """保持比例缩放图片并填充至目标尺寸"""
    shape = img.shape[:2]
    
    # 计算缩放比例
    r = min(target_size[0] / shape[0], target_size[1] / shape[1])
    
    # 计算填充量（居中填充）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (target_size[1] - new_unpad[0]) / 2, (target_size[0] - new_unpad[1]) / 2
    
    # 缩放和填充
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                             cv2.BORDER_CONSTANT, value=[114, 114, 114])

    # 格式转换
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def image_to_bin(image_path, output_bin_path, target_size=(640, 640)):
    """将图片转换为bin文件"""
    # 读取图片，默认读取为BGR格式
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片 {image_path}")
        return False
    
    # 缩放并填充至目标尺寸
    image = letterbox(image, target_size)
    
    # 保存为bin文件
    try:
        image.tofile(output_bin_path)
        return True
    except Exception as e:
        print(f"保存bin文件失败 {output_bin_path}: {e}")
        return False

def process_images(input_dir, output_dir, img_list_file, target_size=(640, 640)):
    """处理输入文件夹中的所有图片，转换为bin文件并保存到输出文件夹"""
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    # 遍历输入文件夹中的所有文件
    with open(img_list_file, 'w') as f_list:
        for filename in os.listdir(input_dir):
            # 检查文件是否为图片
            if filename.lower().endswith(image_extensions):
                input_path = os.path.join(input_dir, filename)
                # 生成输出文件名（保持原名，替换扩展名为.bin）
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}.bin")
                
                # 转换图片为bin文件
                if image_to_bin(input_path, output_path, target_size):
                    print(f"已处理: {filename} -> {base_name}.bin")
                f_list.write(f"img/{base_name}.bin\n")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='将图片批量转换为bin文件')
    
    # 必选参数
    parser.add_argument('--input_dir', required=True, 
                      help='输入图片文件夹路径')
    parser.add_argument('--output_dir', required=True, 
                      help='输出bin文件文件夹路径')
    
    # 可选参数
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                      help='目标尺寸，格式为 宽 高，默认640 640')
    
    args = parser.parse_args()
    
    # 验证输入文件夹是否存在
    if not os.path.isdir(args.input_dir):
        print(f"错误: 输入文件夹 {args.input_dir} 不存在")
        exit(1)
    
    # 处理图片
    process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir + '/img',
        img_list_file=args.output_dir + '/file_list.txt',
        target_size=tuple(args.target_size)
    )
    print(f"处理完成，文件列表已保存至 {args.output_dir + '/file_list.txt'}")

if __name__ == "__main__":
    main()
