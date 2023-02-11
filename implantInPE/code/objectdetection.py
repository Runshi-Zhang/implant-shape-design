import pydicom as dicom
import scipy.ndimage
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mmcv
import numpy as np
from mmdet.apis import init_detector, inference_detector
import os
import cv2
import json

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness
    print([slice_thickness])


    return slices
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16
    image = image.astype(np.int16)

    # sets the outside element to 0
    image[image == -2000] = 0
    # Convert to HU
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope

        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
        image[slice_number] += np.int16(intercept)


    return np.array(image, dtype=np.int16)
def resample(image, scan, new_spacing=[0.5,0.5,0.5]):
    # Determine current pixel spacing
    print(scan[0].PixelSpacing)
    #spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    spacing = np.append([scan[0].SliceThickness], scan[0].PixelSpacing)

    print(image.shape)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    print(real_resize_factor)

    #image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    image = scipy.ndimage.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing, spacing

#def ReData(image,scan,r):



#load the CT data (dicom).
first_patient = load_scan("")
first_patient_pixels = get_pixels_hu(first_patient)

pix_resampled, spacing, oldspacing = resample(first_patient_pixels, first_patient)

# the radius of PCA processing
r = 20
centerIndex = int(pix_resampled.shape[2] / 2)


firpatpix = pix_resampled[:, :, centerIndex - r:centerIndex + r]
pcadata = firpatpix.reshape(firpatpix.shape[0] * firpatpix.shape[1], firpatpix.shape[2])

pca = PCA(n_components=1)  # n_components can be integer or float in (0,1)
pca.fit(pcadata)  # fit the model

pcaResult = pca.fit_transform(pcadata)  # transformed data
re = pcaResult.reshape(firpatpix.shape[0], firpatpix.shape[1])
plt.imshow(re[::-1, ...][:, ::-1], cmap=plt.cm.bone)

imageSize = np.append([512], [512])
print(re.shape)
imageSize = imageSize / re.shape
print(imageSize)

pcaImage = scipy.ndimage.zoom(re, imageSize, mode='nearest')
plt.imsave("./middle.jpg", pcaImage[::-1, ...][:, ::-1], cmap='gray')

#load the object detection file (.py and .pth)
config_file = './cocoTest200.py'
checkpoint_file = './sparsercnn.pth'
model = init_detector(config_file, checkpoint_file)  # or device='cuda:0'
img = mmcv.imread('./middle.jpg')
result = inference_detector(model, img)
plt.imsave("./segimagetruth/" + str(i) + '.jpg',result,
                   cmap='gray')






