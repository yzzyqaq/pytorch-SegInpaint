import numpy as np
import cv2

# 定义标签类
class Label:
    def __init__(self, name, id, trainId, category, catId, hasInstances, ignoreInEval, color):
        self.name = name
        self.id = id
        self.trainId = trainId
        self.category = category
        self.catId = catId
        self.hasInstances = hasInstances
        self.ignoreInEval = ignoreInEval
        self.color = color

# 标签定义
labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]



'''labels_used = []
ids = []
for i in range(len(labels)):
  if(labels[i].ignoreInEval == False):
    labels_used.append(labels[i])
    ids.append(labels[i].id)
 
 
#create a dictionary with label_id as key & train_id as value
label_dic = {}
for i in range(len(labels)-1):
  label_dic[labels[i].id] = labels[i].trainId

print(label_dic)
path = 'gtFine/kin/aachen/aachen_000000_000019_gtFine_color.png'
colored_segmentation_image = cv2.imread(path)
        # 获取图像尺寸
height, width, _ = colored_segmentation_image.shape
label_image = np.zeros((height, width), dtype=np.uint8)
l_un = np.unique(colored_segmentation_image)

print(l_un)
'''

def convert_label(path):
    # 彩色语义分割图
        colored_segmentation_image = cv2.imread(path)


        # 获取图像尺寸
        height, width, _ = colored_segmentation_image.shape

        # 创建空白的标签图
        label_image = np.zeros((height, width), dtype=np.uint8)
        
        # 根据颜色映射生成标签图
        for label in labels:
            # 找到与标签颜色匹配的像素
            mask = np.all(np.abs(colored_segmentation_image - np.array(label.color)) <= 50, axis=-1)
            # 将匹配的像素赋予对应的标签id
            #print(label.trainId)

            label_image[mask] = label.id+1
            print(label_image[mask])
        
        return  label_image



import os
import shutil

def convert_jpg_to_png(folder_path):
    for subdir, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.jpg'):
                jpg_path = os.path.join(subdir, file)
                png_path = os.path.splitext(jpg_path)[0] + '.png'
                shutil.move(jpg_path, png_path)

# 示例用法
folder_path = 'path_to_your_folder'
convert_jpg_to_png(folder_path)


if __name__ == "__main__":
    folder_path = 'E:/gan/spgnet/pytorch-SegInpaint/data\celeba\CelebAMask-HQ/face_parsing\Data_preprocessing\CelebAMask-HQ\CelebA-HQ-img'
    convert_jpg_to_png(folder_path)
    '''input_path = 'visualizations/images/aachen_000173_000019_gtFine_color.png'
     label_semantic_segmentation = convert_label(input_path)
     cv2.imwrite('test/cityscapes/gtFine/train/aachen_000173_000019_gtFine_labelIds.png', label_semantic_segmentation)'''
