"""
Created on Tue Feb 12 14:20 2019

@author: ziatdinovmax
"""

import torch
import numpy as np
import h5py
import cv2


class generate_batches:
    '''Creates a batch generator'''
    
    def __init__(self, hf_file, batch_size, *args, channel_order = 'tfkeras'):
        '''Initializes a `batch generator`'''
        
        self.f = h5py.File(hf_file, 'r')
        self.batch_size = batch_size
        self.channel_order = channel_order
        try:
            self.resize_ = args[0]
        except:
            self.resize_ = None
  
    
    def steps(self, mode = 'training'):
        """Estimates number of steps per epoch"""
        
        if mode == 'val':
            n_samples = self.f['X_test'][:].shape[0]
        else:        
            n_samples = self.f['X_train'][:].shape[0]
        
        return np.arange(n_samples//self.batch_size)
    
    
    def batch(self, idx, mode = 'training'):
        """Generates batch of the selected size
        for training images and the corresponding
        ground truth"""
        
        def batch_resize(X_batch, y_batch, rs):
            '''Resize all images in one batch'''

            if X_batch.shape[1:3] == (rs, rs):
                return X_batch, y_batch
           
            X_batch_r = np.zeros((X_batch.shape[0], rs, rs, X_batch.shape[-1]))
            y_batch_r = np.zeros((y_batch.shape[0], rs, rs)) 
            for i, (img, gt) in enumerate(zip(X_batch, y_batch)):
                img = cv2.resize(img, (rs, rs))
                img = np.expand_dims(img, axis = 2)
                gt = cv2.resize(gt, (rs, rs))
                X_batch_r[i, :, :, :] = img
                y_batch_r[i, :, :] = gt

            return X_batch_r, y_batch_r
        
        if mode == 'val':
            X_batch = self.f['X_test'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
            y_batch = self.f['y_test'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
        else:
            X_batch = self.f['X_train'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
            y_batch = self.f['y_train'][int(idx*self.batch_size):int((idx+1)*self.batch_size)]
        
        if self.resize_ != None:
            rs_arr = np.arange(self.resize_[0], self.resize_[1], self.resize_[2])
            rs = np.random.choice(rs_arr)
            X_batch, y_batch = batch_resize(X_batch, y_batch, rs)
        
        X_batch = torch.from_numpy(X_batch).float()
        if self.channel_order == 'tfkeras':
            X_batch = X_batch.permute(0, 3, 1, 2)
        y_batch = torch.from_numpy(y_batch).long()
  
        yield X_batch, y_batch
            
    
    def close_(self):
        """Closes h5 file"""
        
        if self.f:
            self.f.close()
            self.f = None
            
            
def torch_format(images, norm = 1):
    '''Reshapes dimensions, normalizes (optionally) 
       and converts image data to a pytorch float tensor.
       (assumes mage data is stored as numpy array)'''
    
    if images.ndim == 2:
        images = np.expand_dims(images, axis = 0)
    images = np.expand_dims(images, axis = 1)
    if norm != 0:
        images = (images - np.amin(images))/np.ptp(images)
    images = torch.from_numpy(images).float()
    
    return images

def predict(images, model, gpu = False):
    '''Returns probability of each pixel in image
        belonging to an atom of a particualr type'''
    
    if gpu:
        model.cuda()
        images = images.cuda()
    model.eval()
    with torch.no_grad():
        prob = model.forward(images)
    if gpu:
        model.cpu()
        images.cpu()
    prob = torch.exp(prob)
    prob = prob.permute(0,2,3,1)
    
    return prob
