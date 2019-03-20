import tensorflow as tf
# tf.enable_eager_execution()
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
from tensorflow.keras.layers import Layer,Reshape, Dense, BatchNormalization, Dropout, Flatten,Concatenate, Add
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCHS= 15

def sigmoid(x):
   return 1 / (1 + np.exp(x))

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        'Xp1':tf.VarLenFeature(tf.float32),
        'Xp2': tf.VarLenFeature(tf.float32),
        'ytrue':  tf.VarLenFeature(tf.int64),}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    Xp1 = tf.sparse_tensor_to_dense (parsed_features['Xp1'], default_value=0) 
    Xp2 = tf.sparse_tensor_to_dense (parsed_features['Xp2'], default_value=0)
    ytrue = tf.sparse_tensor_to_dense (parsed_features['ytrue'], default_value=0)

    return Xp1,Xp2,ytrue

def get_pre_dataset(split,batch_size):
    path ='phrase_VGP_{}.tfrecords'.format(split)
    
    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(_parse_function, num_parallel_calls=4)
    
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    Xp1,Xp2,ytrue = iterator.get_next()

    Xp1= tf.reshape(Xp1, [-1,300])
    Xp2= tf.reshape(Xp2, [-1,300])

    ytrue = tf.reshape(ytrue, [-1,1])

    X = [Xp1,Xp2]
    return X,ytrue

def get_length(split):
    df = pd.read_csv('../data/phrase_pair_{}.csv'.format(split),index_col=0)
    return len(df)


def get_input():
    Xp1= Input(shape=(300,),name='Xp1')
    Xp2= Input(shape=(300,),name='Xp2')
    return Xp1,Xp2

def sigmoid_cross_entropy(y_true, y_pred):
    y_true =  K.flatten(y_true)
    y_pred =  K.flatten(y_pred)
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)


def Phrase_Projection_Net(Xp):
    fuse_p = Dense(1000,kernel_initializer=initializers.he_normal())
    fuse = Dense(300,kernel_initializer=initializers.he_normal())

    x = Activation('relu')(BatchNormalization()(fuse_p(Xp)))
    x = Activation('relu')(BatchNormalization()(fuse(x)))

    return x 

def Classifier_Net(x1,x2):

    dense_1= Dense(128,kernel_initializer=initializers.he_normal())
    dense_2= Dense(128,kernel_initializer=initializers.he_normal())
    dense_final = Dense(1,kernel_initializer= initializers.lecun_normal(),activation='sigmoid')

    x = Add()([dense_1(x1),dense_2(x2)])
    x = Activation('relu')(BatchNormalization()(x))
    x = dense_final(Dropout(0.4)(x))
    # x = dense_final(x)

    return x
    


def VGP_Phrase_Only_Model():
    Xp1,Xp2 = get_input()

    x1 = Phrase_Projection_Net(Xp1)
    x2 = Phrase_Projection_Net(Xp2)

    x = Classifier_Net(x1,x2)

    model = Model([Xp1,Xp2],x)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),loss= 'binary_crossentropy',metrics=['acc'])

    return model




def train(model):
    TRAIN_SIZE = get_length('train')
    VAL_SIZE = get_length('val')

    TRAIN_BATCH_SIZE= 256
    VAL_BATCH_SIZE= 256

    TRAIN_STEPS_PER_EPOCH= TRAIN_SIZE // TRAIN_BATCH_SIZE
    VAL_STEPS_PER_EPOCH= VAL_SIZE // VAL_BATCH_SIZE

    X,y = get_pre_dataset('train',TRAIN_BATCH_SIZE)
    X_val,y_val = get_pre_dataset('val',VAL_BATCH_SIZE)

    filepath= 'VGP_Phrase_only.h5'
    path = Path(filepath)
    
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    model.fit(x=X,y=y,validation_data=(X_val,y_val),
                epochs=EPOCHS,
                steps_per_epoch=TRAIN_STEPS_PER_EPOCH,
                validation_steps=VAL_STEPS_PER_EPOCH,
                callbacks=callbacks_list)


    #serialize weights to HDF5
    print("Saved model to disk")

def get_info(split):

    df= pd.read_csv('../data/phrase_pair_{}.csv'.format(split),index_col=0)
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

def evaluating(split,model,batch_size):
    BATCH_SIZE = batch_size
    ytrue = []
    ypred = []
    df,feat_1,feat_2 = get_info(split)
    size = len(df)
    step = size // BATCH_SIZE + 1
    with progressbar.ProgressBar(max_value= step*BATCH_SIZE) as bar:
        k=0
        while k< step*BATCH_SIZE:
            Xp1 = np.zeros((BATCH_SIZE,300))
            Xp2 = np.zeros((BATCH_SIZE,300))

            for j in range(BATCH_SIZE):
                # Load the image
                i= k%size
                Xp1[j] = feat_1[i]
                Xp2[j] = feat_2[i]

                ytrue.append(int(df.ytrue[i]))
                k+=1
                bar.update(k)
            X = [Xp1,Xp2]
            ypred.append(model.predict(X))

    ytrue = ytrue[:size]
    ypred = np.concatenate(ypred,axis=0)
    ypred = ypred.ravel().tolist()
    ypred = ypred[:size]
    return df,ytrue,ypred
    
def evaluate(model):
    path = 'VGP_Phrase_only.h5'
    BATCH_SIZE = 32
    model.load_weights(path)

    val_df,val_ytrue,val_ypred = evaluating('val',model,BATCH_SIZE)

    val_ytrue = np.array(val_ytrue)
    val_ypred = np.array(val_ypred)



    precision, recall, thresholds = precision_recall_curve(val_ytrue, val_ypred)
    f1 = 2 * (precision * recall) / (precision + recall)
    best_ind = np.nanargmax(f1)
    best_threshold = thresholds[best_ind]

    print('prec: %.4f, rec: %.4f, f1: %.4f' % (precision[best_ind],recall[best_ind], f1[best_ind]))
    
    test_df,test_ytrue,test_ypred = evaluating('test',model,BATCH_SIZE)
    
    test_ytrue = np.array(test_ytrue)
    test_ypred = np.array(test_ypred)

    test_ypred = test_ypred > best_threshold
    prec = precision_score(test_ytrue, test_ypred)
    rec = recall_score(test_ytrue, test_ypred)
    f1 = f1_score(test_ytrue, test_ypred)

    print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))
    test_df['ypred'] = test_ypred

    test_df.to_csv('test_phrase_only.csv')



def main():
    parser = argparse.ArgumentParser(
        description='training script for a paraphrase classifier')

    parser.add_argument('--eval', '-e', type=int, default=0)
    

    model = VGP_Phrase_Only_Model()

    args = parser.parse_args()

    
    if args.eval == 0:
        print('training')
        train(model)
    elif args.eval == 1:
        print('evaluating')
        evaluate(model)
    else:
        print('No')
if __name__ == "__main__":
    main()
