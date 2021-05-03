import os
import numpy as np
from pyeddl.tensor import Tensor, DEV_CPU

# customized Data loader for UC3, preprocessed into tensors numpy

class UC3_Dataset():
    def __init__(self, img_directory, mask_directory, num_channels, num_channels_gt, size, is_test=False, transformations=None, gpu = DEV_CPU): #READ DATA
        self.img_directory = img_directory
        ####for the moment, we just use MOL exams
        self.img_list = []
        self.excluded = ['SLG','MOL-001','MOL-060','MOL-061','MOL-062','MOL-063']

        self.test_names = ['MOL-001','MOL-002','MOL-003','MOL-004','MOL-005']
        self.is_test = is_test
        self.img_list = [ i for i in os.listdir(img_directory) if not any(ex in i for ex in self.excluded)]

        if is_test:
           self.img_list = [ i for i in self.img_list if any(ex in i for ex in self.test_names)]
        else:
           self.img_list = [ i for i in self.img_list if all(ex not in i for ex in self.test_names)]

        self.gt_list = ['-'.join(np.delete(i.split('-'),2)) for i in self.img_list ]

        self.mask_directory = mask_directory
        self.num_channels = num_channels
        self.num_channels_gt = num_channels_gt
        self.size = size
        self.transformations = transformations
        
        self.gpu = gpu

    def __getitem__(self, index): # RETURN ONE ITEM ON THE INDEX

        im_frame = np.load(os.path.join(self.img_directory, self.img_list[index])).astype(np.float32)
        im_frame = im_frame/255
        im_frame = (im_frame - 0.2375)/ 0.3159 #[channels,:,:]
        
        
        mask_frame = np.load(os.path.join(self.mask_directory,self.gt_list[index])).astype(np.float32)#unsqueeze(dim = 0)
        mask_frame = mask_frame/255
        if mask_frame.shape[-1] == 1:
          mask_frame = np.moveaxis(mask_frame, [0,1,2], [2,1,0])
        
        return im_frame, mask_frame

    def __len__(self): # RETURN THE DATA LENGTH
        return len(self.img_list)

def LoadBatch(dataset, batch_number , batch_size = 1, permutation=None): 
    
    if permutation is None:
        permutation = np.arange(len(dataset))

    start = batch_number*batch_size
    indexes = permutation[start: start+batch_size]
    gpu = dataset.gpu
    
    x = np.zeros((len(indexes), dataset.num_channels, dataset.size[0], dataset.size[1]))
    y = np.zeros((len(indexes), dataset.num_channels_gt, dataset.size[0], dataset.size[1]))

    for i,idx in enumerate(indexes):
        im_frame, mask_frame = dataset.__getitem__(idx)
        x[i,:,:,:] = im_frame
        y[i,:,:,:] = mask_frame
    
    x = Tensor.fromarray(x, dev=gpu)
    y = Tensor.fromarray(y, dev=gpu)

    return x,y
    

        


