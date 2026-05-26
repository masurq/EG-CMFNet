import os
import cv2
import numpy as np


def find_images_with_pixel_value(folder_path, target_value=6):
    """
    查找文件夹中包含指定像素值的图片

    Args:
        folder_path (str): 图片文件夹路径
        target_value (int): 目标像素值

    Returns:
        list: 包含目标像素值的图片文件名列表
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    # 获取所有图片文件
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in valid_extensions):
            image_files.append(file)

    if not image_files:
        print("没有找到图片文件")
        return []

    print(f"找到 {len(image_files)} 张图片")
    print(f"正在查找包含像素值 {target_value} 的图片...")

    matching_images = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)

        if img is not None:
            # 检查是否包含目标像素值
            if np.any(img == target_value):
                matching_images.append(image_file)
                print(f"找到: {image_file}")

                # 可选：打印该图片中目标像素值的数量
                count = np.sum(img == target_value)
                print(f"  包含 {count} 个像素值为 {target_value} 的像素")

    return matching_images


# 使用示例
if __name__ == "__main__":
    folder_path = "/media/ubuntu/7c2c9e84-2646-43cd-8669-ef651d32fd3b/zm/ZMSeg/run_data/Potsdam_run_best/all_train/gt_no_boundary/"
    target_value = 6

    matching_images = find_images_with_pixel_value(folder_path, target_value)

    print("\n" + "=" * 50)
    if matching_images:
        print(f"找到 {len(matching_images)} 张包含像素值 {target_value} 的图片:")
        for img_name in matching_images:
            print(f"  - {img_name}")
    else:
        print(f"没有找到包含像素值 {target_value} 的图片")