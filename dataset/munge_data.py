import os
import pandas as pd

def mungeFile(img_path):
    classes = os.listdir(img_path)
    image_file = [os.listdir(os.path.join(img_path, class_name)) for class_name in classes]
    image_path_list = []
    for cls_idx, img_list in enumerate(image_file):
        for img in img_list:
            total_path = os.path.join(classes[cls_idx], img)
            image_path_list.append(total_path)

    #target value
    target = []
    for idx, file_list in enumerate(image_file):
        for i in range(len(file_list)):
            target.append(idx)

    
    df = pd.DataFrame({"image_name" : image_path_list, "target" : target}, index=None)

    return df 

if __name__ == "__main__":
    img_path = "C://Datasets/chest_xray/train/"
    train_df = mungeFile(img_path)
    print(train_df)