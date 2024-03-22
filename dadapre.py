import os
import shutil
import os
import shutil

'''def copy_color_images(input_folder, output_folder):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(output_folder, exist_ok=True)

    # 遍历原始文件夹
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            # 检查文件名是否以 '_color.png' 结尾
            if file.endswith('_color.png'):
                # 构建源文件路径
                source_path = os.path.join(root, file)
                
                # 构建目标文件路径
                target_path = os.path.join(output_folder, file)
                
                # 复制文件
                shutil.copy2(source_path, target_path)
                print(f"Copied {file} to {target_path}")
# 替换 'input_folder' 和 'output_folder' 为实际的文件夹路径
input_folder = 'gtFine/val'
output_folder = 'data\cityscapes\seg_pre\seg_val'

copy_color_images(input_folder, output_folder)'''


'''import os
import shutil

# 源文件夹路径
source_folder = 'data\mask\irregular_mask'

# 目标文件夹路径
destination_folder = 'data/cityscapes/irregular_mask'

# 获取b文件夹中的图片文件名列表
b_folder_images = os.listdir('data\cityscapes\seg_pre\seg_train')
print(b_folder_images)

# 创建新的目标文件夹
os.makedirs(destination_folder, exist_ok=True)

# 遍历a文件夹中的图片
for filename in os.listdir(source_folder):
    # 构建新的文件名，使用b文件夹中的图片名称
    new_filename = b_folder_images.pop(0)
    _, file_extension = os.path.splitext(new_filename)
    new_filename = f"{_}_mask{file_extension}"
    new_filepath = os.path.join(destination_folder, new_filename)

    # 拷贝文件到新的目标文件夹并重新命名
    shutil.copy2(os.path.join(source_folder, filename), new_filepath)

print("图片重新命名并保存到新文件夹完成。")'''

'''from PIL import Image
import os

# 源文件夹路径
source_folder = 'test\cityscapes\irregular_mask'

# 目标文件夹路径
destination_folder = 'test'

# 创建目标文件夹
os.makedirs(destination_folder, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 检查文件是否为PNG格式的图像
    if filename.lower().endswith('.png'):
        # 构建源文件和目标文件的完整路径
        a_filepath = os.path.join(source_folder, filename)
        print(a_filepath)
        b_filename = filename.replace('_mask', '')
        #print(b_filename)
        b_filepath = os.path.join(destination_folder, filename)
        print(b_filepath)
       
        with Image.open(a_filepath) as img:
        # 将黑色变为白色，白色变为黑色
            img = img.convert('L').point(lambda x: 255 if x < 128 else 0)

        # 保存处理后的图像到目标文件夹
            img.save(b_filepath)

print("处理完成。")'''



'''from PIL import Image
import os

def resize_images(input_folder, target_size=(256, 256), suffix='_leftImg8bit.png'):
    # 遍历输入文件夹中以特定后缀结尾的图片
    for filename in os.listdir(input_folder):
        if filename.endswith(suffix):
            input_path = os.path.join(input_folder, filename)

            # 打开图像
            img = Image.open(input_path)

            # 调整图像大小
            resized_img = img.resize(target_size, Image.ANTIALIAS)

            # 保存调整大小后的图像（覆盖原始图像）
            resized_img.save(input_path)

if __name__ == "__main__":
    # 输入文件夹路径
    input_folder_path = "E:/gan/spgnet/pytorch-SegInpaint/precess/visualizations"

    # 调整大小的目标尺寸
    target_size = (256, 256)

    # 执行调整大小操作
    resize_images(input_folder_path, target_size)
'''
from PIL import Image
import numpy as np

def apply_mask(original_image_path, mask_image_path, output_image_path):
    # Open original image and mask image
    original_image = Image.open(original_image_path)
    mask_image = Image.open(mask_image_path)

    # Convert mask image to numpy array
    mask_array = np.array(mask_image)

    # Create a black image with the same size as the original image
    black_image = Image.new('RGB', original_image.size, (0, 0, 0))
    white_range = (240, 240, 240)

    # Iterate through each pixel in the mask
    for y in range(mask_array.shape[0]):
        for x in range(mask_array.shape[1]):
            # If the pixel in the mask is white, set the corresponding pixel in the black image to black
            if np.all(mask_array[y, x] >= white_range):
                original_image.putpixel((x, y), (0, 0, 0))

    # Save the resulting image
    original_image.save(output_image_path)

if __name__ == "__main__":
    # 原始图像路径
    original_image_path = "precess/frankfurt_000001_021406_leftImg8bit.png"

    # mask图像路径，白色表示空缺部分
    mask_image_path = "precess/frankfurt_000001_021406_gtFine_mask.png"

    # 输出图像路径
    output_image_path = "visualizations/frankfurt_000001_021406_gtFine_image_marked.png"

    # 执行标记操作
    apply_mask(original_image_path, mask_image_path, output_image_path)

