import os
import pydicom
import numpy as np
from glob import glob
import argparse
from pyecvl import ecvl
import cv2
import time

def win_scale(data, wl, ww, dtype, out_range):
  """
  Scale pixel intensity data using specified window level, width, and intensity range.
  """

  data_new = np.empty(data.shape, dtype=np.double)
  data_new.fill(out_range[1]-1)

  data_new[data <= (wl-ww/2.0)] = out_range[0]
  data_new[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))] = \
       ((data[(data>(wl-ww/2.0))&(data<=(wl+ww/2.0))]-(wl-0.5))/(ww-1.0)+0.5)*(out_range[1]-out_range[0])+out_range[0]
  data_new[data > (wl+ww/2.0)] = out_range[1]-1
  return data_new.astype(dtype)



def dicomtouint8(data):
  rows = int(data[(0x0028,0x0010)].value)
  cols = int(data[(0x0028,0x0011)].value)
  raw_full = data.PixelData
  raw = [raw_full[x] for x in base]
  raw = raw #map(ord, raw)
  converted = ((np.asarray(raw)).reshape(rows, cols))
  adv_raw = [raw_full[x] for x in adv]
  adv_raw = adv_raw #map(ord, adv_raw)
  converted += 256*((np.asarray(adv_raw)).reshape(rows,cols))
  converted = converted.astype(np.double)
  intercept = float(data[(0x0028, 0x1052)].value)
  slope = float(data[(0x0028, 0x1053)].value)
  converted = (slope*converted+intercept)
  converted = win_scale(converted, data[(0x0028, 0x1050)].value, data[(0x0028, 0x1051)].value, np.uint8, [0, 255])
  return converted


root = '/data03/DH_UC3_Brain/'
inputdir = root + 'processed_data_Benninck/*'
#outroot = '/home/deephealth_UC3/dataset_uc3_EL_128/'

aratio = 512
base = list(np.arange(0,524288, 2))
adv = list(np.arange(1,524288, 2))
#rescale_size = 128#512

validation = ['001','002','003','004','005']
#typeofmeasure = 'TTP'

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--target",help='set to TPP,CBF,CBV to process data targets, default None for input', type=str, default=None)
parser.add_argument("--outroot",help='output folder', type=str, default='/home/deephealth_UC3/dataset_uc3_EL_128/')
parser.add_argument("--rescale_size",help='rescale size', type=int, default=128)
args = parser.parse_args()

rescale_size = args.rescale_size

outroot = args.outroot
if not os.path.exists(outroot):
  os.makedirs(outroot)
test_list = glob(inputdir)

test_list = [ x for x in test_list if '_Maps' in x and 'MOL' in x]

#debug
#test_list = [test_list[0]]

start_time = time.time()

for i in test_list:

    print(i)

    if args.target is None:
      typeofmeasure = 'TTP'
    else:
      typeofmeasure = args.target

    all_dicoms_maps = glob(i + '/NLR_'+typeofmeasure+'/*')###all the maps we got

    acq_card = len(glob(i.replace('_Maps', "/*")))

    for map_name in all_dicoms_maps:
      acquisitions = []

      #Do not have dicom metadata
      #map_data = ecvl.DicomRead(map_name)
      map_data = pydicom.dcmread(map_name)

      if args.target is None:
        h = float(map_data[0x20,0x32].value[2])##height

        for id in range(1, acq_card+1):
          data = pydicom.dcmread(i.replace('_Maps', '/Filtered'+(str(id)).rjust(5, '0')+'.dcm'))
          if (float(data[0x20,0x32].value[2])) == h:
            img = cv2.resize(dicomtouint8(data),(rescale_size, rescale_size),interpolation=cv2.INTER_LINEAR)
            acquisitions.append(np.expand_dims(img, axis=0))
        overall_tensor = np.zeros((len(acquisitions), rescale_size, rescale_size),dtype=np.float32)
        for idx in range(len(acquisitions)):
          overall_tensor[idx,:,:] = acquisitions[idx][0,:,:]
      else:
        overall_tensor = np.expand_dims(dicomtouint8(map_data), axis=0)
 
      target_name = map_name
      target_name = target_name.replace( root + 'processed_data_Benninck/', outroot + 'input_tensored/')
      target_name = target_name.replace('.dcm', '.npy')
      
      if args.target is not None:
        target_name = target_name.replace('_Registered_Filtered_3mm_20HU_Maps/NLR_'+typeofmeasure+'/NLR_'+typeofmeasure,'-')
        target_name = target_name.replace('input_tensored', typeofmeasure)
        if not os.path.exists(outroot + typeofmeasure):
            os.makedirs(outroot + typeofmeasure)
      else:
        target_name = target_name.replace('_Registered_Filtered_3mm_20HU_Maps/NLR_'+typeofmeasure+'/NLR_','-{}-'.format(len(acquisitions)))
        target_name = target_name.replace(typeofmeasure,'')
        if not os.path.exists(outroot + 'input_tensored/'):
            os.makedirs(outroot + 'input_tensored/')

      
      if args.target is not None:
        img = ecvl.Image.fromarray(overall_tensor,'cyx', colortype=ecvl.ColorType.GRAY)
        ecvl.ResizeDim(img, img, [rescale_size,rescale_size], interp=ecvl.InterpolationType.linear)
        ecvl.RearrangeChannels(img,img,'cyx')
        #ecvl.ImWrite('test.jpg',img)
        np.save(target_name, np.array(img))
      else:
        np.save(target_name, overall_tensor)

      print(target_name)

print("---Time to Preprocess:\t%s seconds ---" % (time.time() - start_time))

