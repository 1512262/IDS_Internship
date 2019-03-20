import tensorflow as tf
from tensorflow import keras
import pandas as pd
from PIL import Image
import progressbar
import os
import numpy as np
import sys
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_info(split):
    df= pd.read_csv('../data/non_fine-tuned/{}_new_with_ID.csv'.format(split))
    pre_feat= np.load('../data/old_VGP/language/{}.npy'.format(split))
    
    #Using in image tfrecord
    img_path ='../data/new_flickr30k-images/'
    
    p2i_dict = defaultdict(lambda: -1)
    with open('../data/old_VGP/language/{}_uniquePhrases'.format(split)) as f:
        for i, line in enumerate(f):
            p2i_dict[line.rstrip()] = i
    

    file_names = df.image.astype('str').tolist()
    
    len_df= len(df)
    feat_1= np.zeros((len_df,300))
    feat_2= np.zeros((len_df,300))
    with progressbar.ProgressBar(max_value=len_df) as bar:
        for i in range(len_df):
            feat_1[i]= pre_feat[p2i_dict[df.at[i,'phrase1']]]
            feat_2[i]= pre_feat[p2i_dict[df.at[i,'phrase2']]]
            bar.update(i)
    return df,feat_1,feat_2

def write_TFRecords(split):
    TFRecords_file_name= 'VGP_{}.tfrecords'.format(split)
        
    writer = tf.python_io.TFRecordWriter(TFRecords_file_name)
    
    df,feat_1,feat_2 = get_info(split)

    with progressbar.ProgressBar(max_value=len(df)) as bar:
        for i in range(len(df)):
            # Load the image
            id1 = df.at[i,'id_1']
            id2 = df.at[i,'id_2']
            Xp1 = feat_1[i]
            Xp2 = feat_2[i]
            ytrue =df.ytrue[i]
            
            # Create a feature
            feature = {
                'id1': _bytes_feature(tf.compat.as_bytes(id1)),
                'id2': _bytes_feature(tf.compat.as_bytes(id2)),
                'Xp1': _floats_feature(Xp1),
                'Xp2': _floats_feature(Xp2),
                
                'ytrue': _int64_feature(int(ytrue))}
        
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
        
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            
            bar.update(i)
        
    writer.close()
    sys.stdout.flush()

def main():
    for split in ['train','val','test']:
        print(split)
        write_TFRecords(split)

if __name__ == "__main__":
    main()