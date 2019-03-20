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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def get_info(split):
    df= pd.read_csv('../data/old_VGP/{}_new.csv'.format(split))
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
            file_name = str(df.at[i,'image'])+'.npy'
            phrase1 = df.at[i,'phrase1']
            phrase2 = df.at[i,'phrase2']
            Xp1 = feat_1[i]
            Xp2 = feat_2[i]
            
            ddpn_xmin1 = df.at[i,'ddpn_xmin_1']
            ddpn_xmax1 = df.at[i,'ddpn_xmax_1']
            ddpn_ymin1 = df.at[i,'ddpn_ymin_1']
            ddpn_ymax1 = df.at[i,'ddpn_ymax_1']

            ddpn_rois1 = np.array([ddpn_ymin1,ddpn_xmin1,ddpn_ymax1,ddpn_xmax1])

            ddpn_xmin2 = df.at[i,'ddpn_xmin_2']
            ddpn_xmax2 = df.at[i,'ddpn_xmax_2']
            ddpn_ymin2 = df.at[i,'ddpn_ymin_2']
            ddpn_ymax2 = df.at[i,'ddpn_ymax_2']

            ddpn_rois2 = np.array([ddpn_ymin2,ddpn_xmin2,ddpn_ymax2,ddpn_xmax2])

            gt_xmin1 = df.at[i,'gt_xmin_1']
            gt_xmax1 = df.at[i,'gt_xmax_1']
            gt_ymin1 = df.at[i,'gt_ymin_1']
            gt_ymax1 = df.at[i,'gt_ymax_1']

            gt_rois1 = np.array([gt_ymin1,gt_xmin1,gt_ymax1,gt_xmax1])

            gt_xmin2 = df.at[i,'gt_xmin_2']
            gt_xmax2 = df.at[i,'gt_xmax_2']
            gt_ymin2 = df.at[i,'gt_ymin_2']
            gt_ymax2 = df.at[i,'gt_ymax_2']

            gt_rois2 = np.array([gt_ymin2,gt_xmin2,gt_ymax2,gt_xmax2])
            
            ytrue =df.ytrue[i]
            
            # Create a feature
            feature = {
                'file_name': _bytes_feature(tf.compat.as_bytes(file_name)),
                'phrase1': _bytes_feature(tf.compat.as_bytes(phrase1)),
                'phrase2': _bytes_feature(tf.compat.as_bytes(phrase2)),
                'Xp1': _floats_feature(Xp1),
                'Xp2': _floats_feature(Xp2),
                'ddpn_rois1':_floats_feature(ddpn_rois1),
                'ddpn_rois2': _floats_feature(ddpn_rois2),
                'gt_rois1': _floats_feature(gt_rois1),
                'gt_rois2': _floats_feature(gt_rois2),
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