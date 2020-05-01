import pickle
import os
import numpy as np
from matplotlib.image import imread
import tensorflow as tf

def pickleData(mask_path, mask_dest, image_path, image_dest, filenames):
    file_list = open(filenames, 'r')
    lines = file_list.read().splitlines()
    file_list.close()
    
    object_class_colors = np.asarray([[0, 0, 0],
                                [128, 0, 0],
                                [0, 128, 0],
                                [128, 128, 0],
                                [0, 0, 128],
                                [128, 0, 128],
                                [0, 128, 128],
                                [128, 128, 128],
                                [64, 0, 0],
                                [192, 0, 0],
                                [64, 128, 0],
                                [192, 128, 0],
                                [64, 0, 128],
                                [192, 0, 128],
                                [64, 128, 128],
                                [192, 128, 128],
                                [0, 64, 0],
                                [128, 64, 0],
                                [0, 192, 0],
                                [128, 192, 0],
                                [0, 64, 128]])
    
    for j, line in enumerate(lines):
        print(str(np.round(float(1000*(j+1)/len(lines)))/10)+"%", " "+line)
        
        impath = image_path+'/'+line+'.jpg'
        mpath = mask_path+'/'+line+'.png'
        
        imdest = image_dest+'/'+line+'.pkl'
        mdest =  mask_dest+'/'+line+'.pkl'
        
        image = imread(impath)
        mask = imread(mpath)
        
        x = int(16*np.ceil(image.shape[0]/16))
        y = int(16*np.ceil(image.shape[1]/16))
        
        img_padded = np.zeros((x,y,3))
        mask_padded = np.zeros((x,y))
        
        img_padded[:image.shape[0],:image.shape[1],:] = image
        
        for i, label in enumerate(object_class_colors):
            mask_padded[np.where(np.all(mask==label, axis=-1))[:2]]=i
            
        pickle.dump(img_padded, open(imdest,'wb'))
        pickle.dump(mask_padded, open(mdest, 'wb'))
        
        
def genData(img, mask_path, img_path):
    
    impath = img_path+'/'+img+'.pkl'
    mpath = mask_path+'/'+img+'.pkl'
    
    mask_raw = np.load(mpath, allow_pickle=True)
    image_raw = np.load(impath, allow_pickle=True)
    
    onehot = tf.expand_dims(tf.one_hot(mask_raw, depth=21, axis=-1),0)
    image = tf.expand_dims(image_raw.astype(np.float32),0)
    
    return image, onehot
