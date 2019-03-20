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
img_dir = '../data/flickr30k-images/'
EPOCHS= 3

weight_dir = '../VGP/weight/'
csv_dir = '../VGP/csv/'

if os.path.isdir(weight_dir)==False:
    os.mkdir(weight_dir)
if os.path.isdir(csv_dir)==False:
    os.mkdir(csv_dir)

def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
        'file_name': tf.FixedLenFeature([], tf.string),
        'phrase1': tf.FixedLenFeature([], tf.string),
        'phrase2': tf.FixedLenFeature([], tf.string),
        'Xp1':tf.VarLenFeature(tf.float32),
        'Xp2': tf.VarLenFeature(tf.float32),
        'ddpn_rois1': tf.VarLenFeature(tf.float32),
        'ddpn_rois2':tf.VarLenFeature( tf.float32),
        'gt_rois1': tf.VarLenFeature(tf.float32),
        'gt_rois2': tf.VarLenFeature(tf.float32),
        'ytrue':  tf.VarLenFeature(tf.int64),}
    
    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)
    
    file_names = parsed_features['file_name']
    phrase1= parsed_features['phrase1']
    phrase2= parsed_features['phrase2']
    Xp1 = tf.sparse_tensor_to_dense (parsed_features['Xp1'], default_value=0) 
    Xp2 = tf.sparse_tensor_to_dense (parsed_features['Xp2'], default_value=0)

    rois1 = tf.sparse_tensor_to_dense (parsed_features['ddpn_rois1'], default_value=0)
    rois2 = tf.sparse_tensor_to_dense (parsed_features['ddpn_rois2'], default_value=0)

    ytrue = tf.sparse_tensor_to_dense (parsed_features['ytrue'], default_value=0)
    
    Xp1= tf.reshape(Xp1, [300,])
    Xp2= tf.reshape(Xp2, [300,])
    
    rois1= tf.reshape(rois1, [4,])
    rois2= tf.reshape(rois2, [4,])
    
    ytrue = tf.reshape(ytrue, [1,])
    
    return file_names,phrase1,rois1,Xp1,phrase2,rois2,Xp2,ytrue

def get_pre_dataset(split,batch_size):
    path ='VGP_{}.tfrecords'.format(split)
    
    dataset = tf.data.TFRecordDataset(path)

    dataset = dataset.map(_parse_function, num_parallel_calls=32)
    
    dataset = dataset.repeat()
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/gpu:0"))
    return dataset

def get_pretrained_feature(typ,model_typ,list_file_name):
    list_file_name = list_file_name.numpy()
    n = len(list_file_name)
    arr = range(1024,224,-32)
    if model_typ=='densenet':
        Xv = np.zeros((n,14,14,arr[typ]))
    elif model_typ=='vgg':
        Xv = np.zeros((n,14,14,512))

    for i in range(n):
        file_name = list_file_name[i].decode('ascii')
        Xv[i] = np.load('../data/old_VGP/{}_visual_embedding/'.format(model_typ)+str(typ)+'/'+file_name)
    return Xv

def get_image(list_file_name):
    list_file_name = list_file_name.numpy()
    n = len(list_file_name)
    img = np.zeros((n,224,224,3))
    for i in range(n):
        file_name = list_file_name[i].decode('ascii')[:-4]+'.jpg'
        temp = image.load_img(img_dir+file_name, target_size=(224, 224))
        img[i]= image.img_to_array(temp)
    img = preprocess_input(img)
    return img

def get_length(split):
    df = pd.read_csv('../data/old_VGP/{}_new.csv'.format(split),index_col=0)
    return len(df)

def generator_dataset(typ,model_typ,e2e,iterator):
    while True:
        file_names,_,rois1,Xp1,_,rois2,Xp2,ytrue = iterator.get_next()
        if e2e==0:
            Xv = get_pretrained_feature(typ,model_typ,file_names)
        elif e2e==1:
            Xv = get_image(file_names)
        X = [Xv,rois1.numpy(),Xp1.numpy(),rois2.numpy(),Xp2.numpy()]
        y = ytrue.numpy()
        yield (X,y)

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

def get_input(typ,model_typ,e2e):
    if e2e==0:
        if model_typ=='densenet':
            arr = range(1024,224,-32)
            Xv= Input(shape=(14,14,arr[typ]),name='pretrain_feature')
        elif model_typ=='vgg':
            Xv= Input(shape=(14,14,512),name='pretrain_feature')
    elif e2e==1:
        img= Input(shape=(224,224,3),name='image')

    rois1 = Input(shape=(4,),name='rois1')
    rois2 = Input(shape=(4,),name='rois2')

    Xp1= Input(shape=(300,),name='Xp1')
    Xp2= Input(shape=(300,),name='Xp2')
    if e2e==0:
        return Xv,rois1,Xp1,rois2,Xp2
    elif e2e==1:
        return img,rois1,Xp1,rois2,Xp2

def finetune_visual_model(typ,model_typ,e2e,Xv):
    if e2e==0:
        if model_typ=='densenet':
            x = Xv
            if typ==0:
                return Xv
            for i in range(typ):
                x_prev = Lambda(lambda x: x)(x)
                x=BatchNormalization()(Xv)
                x=Activation('relu')(x)
                x=Conv2D(128,(1,1))(x)
                x=BatchNormalization()(x)
                x=Activation('relu')(x)
                x=Conv2D(32,(1,1))(x)
                x=Concatenate()([x_prev,x])
            return x
        if model_typ=='vgg':
            x = Xv
            if typ==0:
                return Xv
            for i in range(typ):
                x=Conv2D(512,(1,1))(x)
            return x
    if e2e==1:
        if model_typ=='densenet':
            densenet = DenseNet121(weights='imagenet',include_top=False,input_shape=(224,224,3))
            pretrained = Model(inputs=densenet.input, outputs=densenet.get_layer(index=308).output)
            del densenet
            return pretrained(Xv)
        elif model_typ=='vgg':
            vgg = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
            pretrained = Model(inputs=vgg.input, outputs=vgg.get_layer(index=17).output)
            del vgg
            return pretrained(Xv)
    
def fusing_model(model_typ,Xvis,Xp):
    if model_typ=='densenet':
        vis_dim=1024
    elif model_typ=='vgg':
        vis_dim=4096

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
    if model_typ=='densenet':
        vis_dim=1024
    elif model_typ=='vgg':
        vis_dim=4096

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

def vgg_fc_model(after_rois):
    Xvis = Flatten()(after_rois)
    Xvis = Dense(units=4096, input_shape=(25088,),kernel_initializer=initializers.he_normal())(Xvis)
    Xvis = Dense(units=4096,kernel_initializer=initializers.he_normal())(Xvis)
    return Xvis

def VGP_model(typ,model_typ,e2e,gated,eval=0):
    if eval==0:
        if e2e==0:
            BATCH_SIZE = 256
        if e2e==1:
            BATCH_SIZE = 64
    elif eval==1:
        BATCH_SIZE = 16
    #Input
    Xv,rois1,Xp1,rois2,Xp2 = get_input(typ,model_typ,e2e)
    data = [Xv,rois1,Xp1,rois2,Xp2]
    #Finetune Visual Embedding
    before_rois = finetune_visual_model(typ,model_typ,e2e,Xv)

    #RoIPooling
    after_rois1 = RoiPooling(pool_size=7,batch_size=BATCH_SIZE,name='roi_pooling_1')([before_rois,rois1])
    after_rois2= RoiPooling(pool_size=7,batch_size=BATCH_SIZE,name='roi_pooling_2')([before_rois,rois2])

    #Pre_Fusing Visual Feature
    if model_typ=='densenet':
        Xvis1 = GlobalAveragePooling2D(name='global_avg_pool_1')(Activation('relu',name='relu_1')(after_rois1))
        Xvis2 = GlobalAveragePooling2D(name='global_avg_pool_2')(Activation('relu',name='relu_2')(after_rois2))
    elif model_typ=='vgg':
        Xvis1 = vgg_fc_model(after_rois1)
        Xvis2 = vgg_fc_model(after_rois2)
    if gated==0:
        #Fusing Feature
        f1 = fusing_model(model_typ,Xvis1,Xp1)
        f2 = fusing_model(model_typ,Xvis2,Xp2)

        #Classifier
        h = classifier_model(f1,f2)
        model = Model(data,h)

    elif gated==1:
        Xv = visual_model(model_typ,Xvis1,Xvis2)
        Xp = phrase_model(Xp1,Xp2)
        h = gated_classifier_model(Xv,Xp)
        model = Model(data,h)

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss= 'binary_crossentropy',
                  metrics=['acc'])
    return model

def train(typ,model_typ,e2e,gated,model):
    TRAIN_SIZE = get_length('train')
    if e2e==0:
        TRAIN_BATCH_SIZE= 256
    if e2e==1:
        TRAIN_BATCH_SIZE= 64
    
    TRAIN_STEPS_PER_EPOCH= TRAIN_SIZE // TRAIN_BATCH_SIZE
    
    iter_train = get_pre_dataset('train',TRAIN_BATCH_SIZE).make_one_shot_iterator()

    history= model.fit_generator(generator_dataset(typ,model_typ,e2e,iter_train),
                                    epochs=EPOCHS,
                                    steps_per_epoch=TRAIN_STEPS_PER_EPOCH)

    #serialize weights to HDF5
    file_name = ''
    if gated==0:
        if e2e==0:
            file_name = 'VGP_{}_model_type'.format(model_typ)+str(typ)+'.h5'
        if e2e==1:
            file_name = 'VGP_{}_model_e2e'.format(model_typ)+'.h5'
    elif gated==1:
        if e2e==0:
            file_name = 'VGP_{}_model_type_with_gated'.format(model_typ)+str(typ)+'.h5'
        if e2e==1:
            file_name = 'VGP_{}_model_e2e_with_gated'.format(model_typ)+'.h5'
    
    file_path = weight_dir+file_name
    model.save_weights(file_path)
    print("Saved model to disk")

def get_info(split):
    df= pd.read_csv('../data/old_VGP/{}_new.csv'.format(split),index_col=0)
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

def evaluating(typ,model_typ,e2e,split,model):
    BATCH_SIZE = 16
    ytrue = []
    ypred = []
    df,feat_1,feat_2 = get_info(split)
    size = len(df)
    step = size // BATCH_SIZE + 1
    with progressbar.ProgressBar(max_value= step*BATCH_SIZE) as bar:
        k=0
        arr = range(1024,224,-32)
        while k< step*BATCH_SIZE:
            if e2e==0:
                if model_typ=='densenet':
                    Xv = np.zeros((BATCH_SIZE,14,14,arr[typ]))
                elif model_typ=='vgg':
                    Xv = np.zeros((BATCH_SIZE,14,14,512))
            elif e2e==1:
                Xv = np.zeros((BATCH_SIZE,224,224,3))
            Xp1 = np.zeros((BATCH_SIZE,300))
            Xp2 = np.zeros((BATCH_SIZE,300))
            rois1 = np.zeros((BATCH_SIZE,4))
            rois2 = np.zeros((BATCH_SIZE,4))

            for j in range(BATCH_SIZE):
                # Load the image
                i= k%size
                file_name = str(df.at[i,'image'])
                if e2e==0:
                    file_name+='.npy'
                    Xv[j]= np.load('../data/old_VGP/{}_visual_embedding/'.format(model_typ)+str(typ)+'/'+file_name)
                if e2e==1:
                    file_name+='.jpg'
                    img = image.load_img(img_dir+file_name, target_size=(224, 224))
                    Xv[j]= image.img_to_array(img)
                    
                Xp1[j] = feat_1[i]
                Xp2[j] = feat_2[i]

                ddpn_xmin1 = df.at[i,'ddpn_xmin_1']
                ddpn_xmax1 = df.at[i,'ddpn_xmax_1']
                ddpn_ymin1 = df.at[i,'ddpn_ymin_1']
                ddpn_ymax1 = df.at[i,'ddpn_ymax_1']

                rois1[j] = np.array([ddpn_ymin1,ddpn_xmin1,ddpn_ymax1,ddpn_xmax1])

                ddpn_xmin2 = df.at[i,'ddpn_xmin_2']
                ddpn_xmax2 = df.at[i,'ddpn_xmax_2']
                ddpn_ymin2 = df.at[i,'ddpn_ymin_2']
                ddpn_ymax2 = df.at[i,'ddpn_ymax_2']

                rois2[j] = np.array([ddpn_ymin2,ddpn_xmin2,ddpn_ymax2,ddpn_xmax2])
                ytrue.append(int(df.ytrue[i]))
                k+=1
                bar.update(k)
            if e2e==1:
                Xv = preprocess_input(Xv)
            X = [Xv,rois1,Xp1,rois2,Xp2]
            ypred.append(model.predict(X))

    ytrue = ytrue[:size]
    ypred = np.concatenate(ypred,axis=0)
    ypred = ypred.ravel().tolist()
    ypred = ypred[:size]
    return df,ytrue,ypred
    
def evaluate(typ,model_typ,e2e,gated,model):
    if gated==0:
        if e2e==0:
            file_name = 'VGP_{}_model_type'.format(model_typ)+str(typ)+'.h5'
        if e2e==1:
            file_name = 'VGP_{}_model_e2e'.format(model_typ)+'.h5'
    elif gated==1:
        if e2e==0:
            file_name = 'VGP_{}_model_type_with_gated'.format(model_typ)+str(typ)+'.h5'
        if e2e==1:
            file_name = 'VGP_{}_model_e2e_with_gated'.format(model_typ)+'.h5'
            
    file_path = weight_dir+file_name
    
    model.load_weights(file_path)

    val_df,val_ytrue,val_ypred = evaluating(typ,model_typ,e2e,'val',model)

    val_ytrue = np.array(val_ytrue)
    val_ypred = np.array(val_ypred)

    precision, recall, thresholds = precision_recall_curve(val_ytrue, val_ypred)
    f1 = 2 * (precision * recall) / (precision + recall)
    best_ind = np.nanargmax(f1)
    best_threshold = thresholds[best_ind]

    print('prec: %.4f, rec: %.4f, f1: %.4f' % (precision[best_ind],recall[best_ind], f1[best_ind]))
    
    test_df,test_ytrue,test_ypred = evaluating(typ,model_typ,e2e,'test',model)
    
    test_ytrue = np.array(test_ytrue)
    test_ypred = np.array(test_ypred)

    test_ypred = test_ypred > best_threshold
    prec = precision_score(test_ytrue, test_ypred)
    rec = recall_score(test_ytrue, test_ypred)
    f1 = f1_score(test_ytrue, test_ypred)

    print('prec: %.4f, rec: %.4f, f1: %.4f' % (prec, rec, f1))
    test_df['ypred'] = test_ypred

    csv_name = ''
    if gated==0:
        if e2e==0:
            csv_name = 'VGP_{}_model_type'.format(model_typ)+str(typ)+'.csv'
        if e2e==1:
            csv_name = 'VGP_{}_model_e2e'.format(model_typ)+'.csv'
    elif gated==1:
        if e2e==0:
            csv_name = 'VGP_{}_model_type_with_gated'.format(model_typ)+str(typ)+'.csv'
        if e2e==1:
            csv_name = 'VGP_{}_model_e2e_with_gated'.format(model_typ)+'.csv'

    csv_path = csv_dir+csv_name
    test_df.to_csv(csv_name)

def main():
    parser = argparse.ArgumentParser(
        description='training script for a paraphrase classifier')

    parser.add_argument('--eval', '-e', type=int, default=0)
    parser.add_argument('--type', '-t', type=int, default=0)
    parser.add_argument('--model', '-m', type=str, default='densenet')
    parser.add_argument('--gated', '-g', type=int, default=0)
    parser.add_argument('--e2e', type=int, default=0)

    args = parser.parse_args()

    print('Model:',args.model)

    if args.gated == 1:
        print('with Gated')

    if args.e2e == 0:
        print('Finetune:',args.type,'layers')
    elif args.e2e == 1:
        print('End-to-End')
    
    if args.eval == 0:
        print('training')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.type%4)
        model = VGP_model(args.type,args.model,args.e2e,args.gated,args.eval)
        train(args.type,args.model,args.e2e,args.gated,model)
    elif args.eval == 1:
        print('evaluating')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.type%4)
        model = VGP_model(args.type,args.model,args.e2e,args.gated,args.eval)
        evaluate(args.type,args.model,args.e2e,args.gated,model)
    else:
        print('No')
if __name__ == "__main__":
    main()
