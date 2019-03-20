import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.layers import Activation,GlobalAveragePooling2D,Lambda, Reshape,Conv2D
from tensorflow.keras.layers import Layer,Reshape, Dense, BatchNormalization, Dropout, Flatten,Concatenate, Add, Multiply
from tensorflow.keras import Model, Input, layers, initializers, backend, Sequential
from tensorflow.keras.utils import plot_model
import progressbar
import pandas as pd
from PIL import Image
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
BATCH_SIZE = 1

def get_input():
    img= Input(shape=(224,224,3),name='image')
    rois = Input(shape=(4,),name='rois')
    return img,rois

class RoiPooling(Layer):
    def __init__(self, pool_size, batch_size, **kwargs):
        self.pool_size = pool_size
        self.batch_size = batch_size

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]
        super(RoiPooling, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([None, self.pool_size, self.pool_size, self.nb_channels])

    def call(self, x, mask=None):
        assert(len(x) == 2)

        img = x[0]
        rois = x[1]
        input_shape = tf.shape(img)
        outputs = []

        for roi_idx in range(self.batch_size):

            ymin = rois[roi_idx, 0] * 14 / 224
            xmin = rois[roi_idx, 1] * 14 / 224
            ymax = rois[roi_idx, 2] * 14 / 224
            xmax = rois[roi_idx, 3] * 14 / 224
            
            xmin = tf.cast(xmin, 'int32')
            ymin = tf.cast(ymin, 'int32')
            xmax = tf.cast(xmax, 'int32')
            ymax = tf.cast(ymax, 'int32')
            
            feature_map = img[roi_idx, ymin:ymax+1, xmin:xmax+1, :]
            post_feat = tf.expand_dims(feature_map,0)

            rs = tf.image.resize_images(post_feat, (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = tf.concat(outputs, axis=0)
        final_output = tf.reshape(final_output, (self.batch_size, self.pool_size, self.pool_size, self.nb_channels))

        final_output = tf.transpose(final_output, perm=[0, 1, 2, 3])
        return final_output

def visual_model():
    img,rois = get_input()
    vgg = VGG16(weights='imagenet',include_top=True,input_shape=(224,224,3))
    fc1_weight = vgg.get_layer('fc1').get_weights()
    fc2_weight = vgg.get_layer('fc2').get_weights()

    pretrained = Model(inputs=vgg.input, outputs=vgg.get_layer(index=17).output)
    before_rois = pretrained(img)
    after_rois = RoiPooling(pool_size=7,batch_size=BATCH_SIZE,name='roi_pooling_1')([before_rois,rois])
    x = Flatten()(after_rois)
    x = Dense(units=4096, input_shape=(25088,),name='fc1')(x)
    x = Dense(units=4096,name='fc2')(x)
    model = Model([img,rois],x)
    model.get_layer('fc1').set_weights(fc1_weight)
    model.get_layer('fc2').set_weights(fc2_weight)
    del vgg
    return model

def prepair_data(split):
    df = pd.read_csv('../data/non_fine-tuned/{}_new_with_ID.csv'.format(split))
    list_id = list(set(df.id_1.tolist()+ df.id_2.tolist()))
    f = open('../data/non_fine-tuned/{}_uniqueID'.format(split),'w')
    for idx in list_id:
        f.write(idx+'\n')
    f.close()
    return list_id

def extract_info(ch):
    words = ch.split('_')
    file_name = words[0]
    rois = np.array([int(words[1]),int(words[2]),int(words[3]),int(words[4])])

    return file_name, rois

def extract_feature(split):
    model = visual_model()
    img_dir='../data/flickr30k-images/'
    emb_dir='../data/non_fine-tuned/vgg_visual_embedding/'+split+'/'
    # if os.path.isdir(emb_dir)==False:
    #     os.mkdir(emb_dir[-1])
    
    list_id = prepair_data(split)
    with progressbar.ProgressBar(max_value=len(list_id)) as bar:
        for i in range(len(list_id)):
            file_name, rois = extract_info(list_id[i])

            img_path = img_dir + file_name +'.jpg'
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            r = np.zeros((1,4))
            r[0] = rois
            Xvis = model.predict([x,r])
            npy_file_name = emb_dir + list_id[i] 
            np.save(npy_file_name,np.squeeze(Xvis))
            bar.update(i)


def main():
    for split in ['train','val','test']:
        print(split)
        extract_feature(split)

if __name__ == "__main__":
    main()
