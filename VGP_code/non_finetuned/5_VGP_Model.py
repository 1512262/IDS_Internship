import tensorflow as tf
tf.enable_eager_execution()
import pandas as pd
from tensorflow import keras
import numpy as np
import pandas as pd
import json
from tensorflow.keras.preprocessing import image
from pathlib import Path
import os
from collections import defaultdict
from tensorflow.keras.layers import Activation,GlobalAveragePooling2D,Lambda, Reshape,Conv2D
from tensorflow.keras.layers import Layer,Reshape, Dense, BatchNormalization, Dropout, Flatten,Concatenate, Add, Multiply
from tensorflow.keras import Model, Input, layers, initializers, backend, Sequential
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,LearningRateScheduler
from sklearn.metrics import f1_score, precision_score, recall_score,precision_recall_curve
from tensorflow.keras.utils import multi_gpu_model
from PIL import Image
from datetime import datetime
from tensorflow.python import debug as tf_debug
import progressbar
import argparse
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

img_dir = '../data/flickr30k-images/'
EPOCHS= 5

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        'id1': tf.FixedLenFeature([], tf.string),
        'id2': tf.FixedLenFeature([], tf.string),
        'Xp1':tf.VarLenFeature(tf.float32),
        'Xp2': tf.VarLenFeature(tf.float32),
        'ytrue':  tf.VarLenFeature(tf.int64),}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    id1= parsed_features['id1']
    id2= parsed_features['id2']
    Xp1 = tf.sparse_tensor_to_dense (parsed_features['Xp1'], default_value=0) 
    Xp2 = tf.sparse_tensor_to_dense (parsed_features['Xp2'], default_value=0)

    ytrue = tf.sparse_tensor_to_dense (parsed_features['ytrue'], default_value=0)
    
    Xp1= tf.reshape(Xp1, [300,])
    Xp2= tf.reshape(Xp2, [300,])
    
    ytrue = tf.reshape(ytrue, [1,])
    
    return id1,Xp1,id2,Xp2,ytrue

def get_pre_dataset(split,batch_size):
    path ='VGP_{}.tfrecords'.format(split)
    
    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(_parse_function, num_parallel_calls=32)
    
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
    return dataset

def get_pretrained_feature(model_typ,split,list_file_name):
    list_file_name = list_file_name.numpy()
    n = len(list_file_name)
    if model_typ=='vgg':
        Xv = np.zeros((n,4096))
    elif model_typ=='densenet':
        Xv = np.zeros((n,1024))
    for i in range(n):
        file_name = list_file_name[i].decode('ascii')
        Xv[i] = np.load('../data/non_fine-tuned/'+model_typ+'_visual_embedding/{}/'.format(split)+file_name+'.npy')
    return Xv

def get_length(split):
    df = pd.read_csv('../data/non_fine-tuned/{}_new_with_ID.csv'.format(split),index_col=0)
    return len(df)

def generator_dataset(model_typ,iterator):
    while True:
        id1,Xp1,id2,Xp2,ytrue = iterator.get_next()
        Xv1= get_pretrained_feature(model_typ,'train',id1)
        Xv2= get_pretrained_feature(model_typ,'train',id2)
        X = [Xv1,Xp1.numpy(),Xv2,Xp2.numpy()]
        y = ytrue.numpy()
        yield (X,y)

def get_input(model_typ):
    if model_typ=='vgg':
        vis_dim=4096
    elif model_typ=='densenet':
        vis_dim=1024

    Xv1 = Input(shape=(vis_dim,),name='pretrain_feature_1')
    Xv2 = Input(shape=(vis_dim,),name='pretrain_feature_2')

    Xp1= Input(shape=(300,),name='Xp1')
    Xp2= Input(shape=(300,),name='Xp2')
    return Xv1,Xp1,Xv2,Xp2
 
def fusing_model(model_typ,Xvis,Xp):
    if model_typ=='vgg':
        vis_dim=4096
    elif model_typ=='densenet':
        vis_dim=1024

    dense_Xvis= Dense(units=1000, input_shape=(vis_dim,),kernel_initializer=initializers.he_normal())
    dense_Xfuse_v= Dense(units=1000, input_shape=(1000,),kernel_initializer=initializers.he_normal())
    dense_Xfuse_p= Dense(units=1000, input_shape=(300,),kernel_initializer=initializers.he_normal())
    dense_fuse= Dense(units=300, input_shape=(1000,),kernel_initializer=initializers.he_normal())

    Xvis = Activation('relu')(BatchNormalization()((dense_Xvis(Xvis))))
    fuse = Add()([dense_Xfuse_v(Xvis),dense_Xfuse_p(Xp)])

    fuse= Activation('relu')(BatchNormalization()(fuse))
    fuse= Activation('relu')(BatchNormalization()(dense_fuse(fuse)))
    return fuse

def classifier_model(f1,f2):
    mlp_1= Dense(128,kernel_initializer= initializers.he_normal(),name='mlp_1')
    mlp_2= Dense(128,kernel_initializer= initializers.he_normal(),name='mlp_2')
    final_mlp = Dense(1,kernel_initializer= initializers.lecun_normal(),activation='sigmoid',name='final')

    final_fuse = Add()([mlp_1(f1),mlp_2(f2)])

    h= Activation('relu')(BatchNormalization()(final_fuse))
    h= final_mlp(Dropout(0.4)(h))
    h= Flatten()(h)
    return h

def multimodal_gate_model(Xv,Xp):
    dense_phr = Dense(300,kernel_initializer= initializers.he_normal())
    dense_vis = Dense(300,kernel_initializer= initializers.he_normal())

    x = Concatenate()([Xv,Xp])
    g_phr = Activation('sigmoid')(dense_phr(x))
    g_vis = Activation('sigmoid')(dense_vis(x))
    return g_phr,g_vis

def gated_classifier_model(Xv,Xp):
    mlp_phr= Dense(300,kernel_initializer= initializers.he_normal())
    mlp_vis= Dense(300,kernel_initializer= initializers.he_normal())
    final_mlp = Dense(1,kernel_initializer= initializers.lecun_normal(),activation='sigmoid',name='final')

    g_phr,g_vis = multimodal_gate_model(Xv,Xp)
    h_phr= Activation('tanh')(BatchNormalization()(mlp_phr(Xp)))
    h_vis= Activation('tanh')(BatchNormalization()(mlp_vis(Xv)))

    h = Add()([Multiply()([g_phr,h_phr]), Multiply()([g_vis,h_vis])])
    h= final_mlp(Dropout(0.4)(h))
    h= Flatten()(h)
    return h

def visual_model(model_typ,Xv1,Xv2):
    if model_typ=='vgg':
        vis_dim=4096
    elif model_typ=='densenet':
        vis_dim=1024

    dense_Xvis_1= Dense(units=1000, input_shape=(vis_dim,),kernel_initializer=initializers.he_normal())
    dense_x1 = Dense(units=1000, input_shape=(1000,),kernel_initializer=initializers.he_normal())
    dense_Xvis_2= Dense(units=1000, input_shape=(vis_dim,),kernel_initializer=initializers.he_normal())
    dense_x2 = Dense(units=1000, input_shape=(1000,),kernel_initializer=initializers.he_normal())

    x1 = Activation('relu')(BatchNormalization()(dense_Xvis_1(Xv1)))
    x2 = Activation('relu')(BatchNormalization()(dense_Xvis_2(Xv2)))
    x = Add()([dense_x1(x1),dense_x2(x2)])
    Xv = Activation('relu')(BatchNormalization()(x))
    return Xv

def phrase_model(Xp1,Xp2):
    dense_Xp_1= Dense(units=1000, input_shape=(300,),kernel_initializer=initializers.he_normal())
    dense_x1 = Dense(units=1000, input_shape=(1000,),kernel_initializer=initializers.he_normal())
    dense_Xp_2= Dense(units=1000, input_shape=(300,),kernel_initializer=initializers.he_normal())
    dense_x2 = Dense(units=1000, input_shape=(1000,),kernel_initializer=initializers.he_normal())

    x1 = Activation('relu')(BatchNormalization()(dense_Xp_1(Xp1)))
    x2 = Activation('relu')(BatchNormalization()(dense_Xp_2(Xp2)))
    x = Add()([dense_x1(x1),dense_x2(x2)])
    Xp = Activation('relu')(BatchNormalization()(x))
    return Xp



def VGP_model(model_typ,gated,eval=0):
    if eval==0:
        BATCH_SIZE = 256
    elif eval==1:
        BATCH_SIZE = 16
    #Input
    Xv1,Xp1,Xv2,Xp2 = get_input(model_typ)
    data = [Xv1,Xp1,Xv2,Xp2]

    if gated==0:
        #Fusing Feature
        f1 = fusing_model(model_typ,Xv1,Xp1)
        f2 = fusing_model(model_typ,Xv2,Xp2)

        #Classifier
        h = classifier_model(f1,f2)
        model = Model(data,h)

    elif gated==1:
        Xv = visual_model(model_typ,Xv1,Xv2)
        Xp = phrase_model(Xp1,Xp2)
        h = gated_classifier_model(Xv,Xp)
        model = Model(data,h)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss= 'binary_crossentropy',
                  metrics=['acc'])
    return model

def train(model_typ,gated,model):
    TRAIN_SIZE = get_length('train')
    TRAIN_BATCH_SIZE= 256
    
    TRAIN_STEPS_PER_EPOCH= TRAIN_SIZE // TRAIN_BATCH_SIZE
    
    iter_train = get_pre_dataset('train',TRAIN_BATCH_SIZE).make_one_shot_iterator()

    history= model.fit_generator(generator_dataset(model_typ,iter_train),
                                    epochs=EPOCHS,
                                    steps_per_epoch=TRAIN_STEPS_PER_EPOCH)

    #serialize weights to HDF5
    if gated==0:
        file_name='VGP_Model_{}.h5'.format(model_typ)
    elif gated==1:
        file_name='VGP_Model_with_gated_{}.h5'.format(model_typ)
    
    model.save_weights(file_name)
    print("Saved model to disk")

def get_info(split):
    df= pd.read_csv('../data/non_fine-tuned/{}_new_with_ID.csv'.format(split),index_col=0)
    pre_feat= np.load('../data/old_VGP/language/{}.npy'.format(split))

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

def evaluating(model_typ,split,model):
    BATCH_SIZE = 16
    if model_typ=='vgg':
        vis_dim=4096
    elif model_typ=='densenet':
        vis_dim=1024
    ytrue = []
    ypred = []
    df,feat_1,feat_2 = get_info(split)
    size = len(df)
    step = size // BATCH_SIZE + 1
    with progressbar.ProgressBar(max_value= step*BATCH_SIZE) as bar:
        k=0
        while k< step*BATCH_SIZE:

            Xv1 = np.zeros((BATCH_SIZE,vis_dim))
            Xv2 = np.zeros((BATCH_SIZE,vis_dim))
            Xp1 = np.zeros((BATCH_SIZE,300))
            Xp2 = np.zeros((BATCH_SIZE,300))

            for j in range(BATCH_SIZE):
                # Load the image
                i= k%size
                id1 = str(df.at[i,'id_1'])
                id2 = str(df.at[i,'id_2'])
                Xv1[j]= np.load('../data/non_fine-tuned/'+model_typ+'_visual_embedding/{}/'.format(split)+id1+'.npy')
                Xv2[j]= np.load('../data/non_fine-tuned/'+model_typ+'_visual_embedding/{}/'.format(split)+id2+'.npy')
                    
                Xp1[j] = feat_1[i]
                Xp2[j] = feat_2[i]

                ytrue.append(int(df.ytrue[i]))
                k+=1
                bar.update(k)
            X = [Xv1,Xp1,Xv2,Xp2]
            ypred.append(model.predict(X))

    ytrue = ytrue[:size]
    ypred = np.concatenate(ypred,axis=0)
    ypred = ypred.ravel().tolist()
    ypred = ypred[:size]
    return df,ytrue,ypred
    
def evaluate(model_typ,gated,model):
    if gated==0:
        file_name='VGP_Model_{}.h5'.format(model_typ)
    elif gated==1:
        file_name='VGP_Model_with_gated_{}.h5'.format(model_typ)
    
    model.load_weights(file_name)

    val_df,val_ytrue,val_ypred = evaluating(model_typ,'val',model)

    val_ytrue = np.array(val_ytrue)
    val_ypred = np.array(val_ypred)

    precision, recall, thresholds = precision_recall_curve(val_ytrue, val_ypred)
    f1 = 2 * (precision * recall) / (precision + recall)
    best_ind = np.nanargmax(f1)
    best_threshold = thresholds[best_ind]

    print('prec: %.4f, rec: %.4f, f1: %.4f' % (precision[best_ind],recall[best_ind], f1[best_ind]))
    
    test_df,test_ytrue,test_ypred = evaluating(model_typ,'test',model)
    
    test_ytrue = np.array(test_ytrue)
    test_ypred = np.array(test_ypred)

    test_ypred = test_ypred > best_threshold
    prec = precision_score(test_ytrue, test_ypred)
    rec = recall_score(test_ytrue, test_ypred)
    f1 = f1_score(test_ytrue, test_ypred)

    print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))
    test_df['ypred'] = test_ypred

    if gated==0:
        csv_name='test_VGP_Model_{}.csv'.format(model_typ)
    elif gated==1:
        csv_name='test_VGP_Model_with_gated_{}.csv'.format(model_typ)

    test_df.to_csv(csv_name)

def main():
    parser = argparse.ArgumentParser(
        description='training script for a paraphrase classifier')

    parser.add_argument('--eval', '-e', type=int, default=0)
    parser.add_argument('--gated', '-g', type=int, default=0)
    parser.add_argument('--model_type', '-m', type=str, default='densenet')

    args = parser.parse_args()
    print(args.model_type)
    if args.gated == 1:
        print('with Gated')

    if args.eval == 0:
        print('training')
        model = VGP_model(args.model_type,args.gated,args.eval)
        train(args.model_type,args.gated,model)
    elif args.eval == 1:
        print('evaluating')
        model = VGP_model(args.model_type,args.gated,args.eval)
        evaluate(args.model_type,args.gated,model)
    else:
        print('No')
if __name__ == "__main__":
    main()
