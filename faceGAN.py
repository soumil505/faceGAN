# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:34:56 2018

@author: soumil
"""

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import cv2
import os
import matplotlib.pyplot as plt
import datetime
from imutils import face_utils
import glob
import numpy as np


class GAN():
    def __init__(self,face_dir,landmark_dir):
        self.face_dir=face_dir
        self.landmark_dir=landmark_dir
        self.faces=sorted(glob.glob(face_dir+'/*.png'))[:100]
        self.landmarks=sorted(glob.glob(landmark_dir+'/*.png'))[:100]
        self.img_shape=(256,256,3)
        self.batch_size=1
        self.n_batches=int(len(self.faces)/self.batch_size)
        self.channels=3
        self.gf=32 #if only
        self.df=32
        self.dataset_name="soumil_face"
        
        
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.generator = self.build_generator()
        self.generator.summary()
        face=Input(self.img_shape)
        landmark=Input(shape=(self.img_shape[0],self.img_shape[1],1))
        
        fake_face=self.generator(landmark)
        self.discriminator.trainable = False
        
        validity = self.discriminator([fake_face,landmark])
        self.combined = Model(inputs=[face,landmark], outputs=[validity, fake_face])
        self.combined.summary()
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)
        
        
        
    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        

        img_inp=Input(shape=self.img_shape,name="discriminator_img_input")
        l_inp=Input(shape=(self.img_shape[0],self.img_shape[1],1),name="discriminator_exxpression_input")
        combined_imgs = Concatenate(axis=-1)([img_inp, l_inp])
        
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        d5 = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        validity=Dense(1,activation='sigmoid')(Flatten()(d5))
        model=Model([img_inp,l_inp],validity)
        return model

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=(self.img_shape[0],self.img_shape[1],1),name="generator_input")

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)

        # Upsampling
        u3 = deconv2d(d5, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)
    def load_batch(self, is_testing=False):
        imgs_A=np.zeros((self.n_batches,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        imgs_B=np.zeros((self.n_batches,self.img_shape[0],self.img_shape[1],self.img_shape[2]))
        for i in range(self.n_batches-1):
            f_batch = self.faces[i*self.batch_size:(i+1)*self.batch_size]
            l_batch = self.landmarks[i*self.batch_size:(i+1)*self.batch_size]
            
            
            for img1,img2 in zip(f_batch,l_batch):
                img_A = cv2.imread(img1)
                img_B = cv2.imread(img2)
                imgs_A[i,:,:,:]=img_A
                imgs_B[i,:,:,:]=img_B
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.
        yield imgs_A, imgs_B

    def train(self, epochs, sample_interval=10):
        valid = np.ones((self.n_batches,1))
        fake = np.zeros((self.n_batches,1))
        start_time = datetime.datetime.now()
        for epoch in range(epochs):
            for batch_i, (img_faces, img_landmarks) in enumerate(self.load_batch()):
                img_landmarks=np.reshape(img_landmarks[:,:,:,0],(-1,self.img_shape[0],self.img_shape[1],1))
                fake_face = self.generator.predict(img_landmarks)
                d_loss_real = self.discriminator.train_on_batch([img_faces, img_landmarks], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_face, img_landmarks], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                
                
                print(np.shape(img_faces))
                print(np.shape(img_landmarks))
                g_loss = self.combined.train_on_batch([img_faces, img_landmarks], [valid, img_faces])
                elapsed_time = datetime.datetime.now() - start_time
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
    def load_data(self):
        index=np.random.randint(0,len(self.faces))
        face_images = [self.faces[index]]
        l_images= [self.landmarks[index]]
        imgs_A=[]
        imgs_B=[]
        for f,l in zip(face_images,l_images):
            img_A=cv2.imread(f)
            img_B=cv2.imread(l)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def sample_images(self,epoch,batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 1, 3
        imgs_A, imgs_B = self.load_data()
        imgs_B=np.reshape(imgs_B[:,:,:,0],(-1,self.img_shape[0],self.img_shape[1],1))
        fake_A = self.generator.predict(imgs_B)
        l_img=np.zeros(np.shape(imgs_A))
        print(np.shape(l_img))
        print(np.shape(imgs_B))
        l_img[:,:,:,0]=imgs_B[:,:,:,0]
        gen_imgs = np.concatenate([l_img, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        plt.subplot(r, c, 1)
        plt.imshow(l_img)
        plt.subplot(r, c, 2)
        plt.imshow(fake_A)
        plt.subplot(r, c, 3)
        plt.imshow(imgs_A)
        
        fig=plt.gcf()
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()
            
            
            
        
        
        
if __name__=='__main__':
    model=GAN('./train_img','./train_label')
    model.train(epochs=200)