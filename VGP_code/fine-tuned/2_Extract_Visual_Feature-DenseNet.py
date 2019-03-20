import os
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import DenseNet121,preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import progressbar
from PIL import Image
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 


def get_pretrained_model(type):
    arr = range(308,133,-7)
    densenet = DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3))
    pretrained = Model(inputs=densenet.input, outputs=densenet.get_layer(index=arr[type]).output)
    del densenet
    return pretrained

def main():
    
    parser = argparse.ArgumentParser(description='training script for a paraphrase classifier')

    parser.add_argument('--type', '-t', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.type%4)
    print('DenseNet-121 for finetune',args.type,'layers')
    print('Using GPU',args.type)
    pretrained = get_pretrained_model(args.type)

    img_dir='../data/flickr30k-images/'
    emb_dir='../data/old_VGP/densenet_visual_embedding/'+str(args.type)+'/'
    if not os.path.exists(emb_dir):
        os.makedirs(emb_dir)
    list_file= os.listdir(img_dir)
    img_path_list = [img_dir+file_name for file_name in list_file]

    i=0
    with progressbar.ProgressBar(max_value=len(img_path_list)) as bar:
        for img_path in img_path_list:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            preds = pretrained.predict(x)
            npy_file_name = emb_dir+img_path[len(img_dir):-4]
            np.save(npy_file_name,np.squeeze(preds))
            i+=1
            bar.update(i)

if __name__ == "__main__":
    main()
