import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def img_to_ascii(
    path_to_image, components=20, resize=(None, None), 
    ascii_str='-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm'):
    '''Convert image to ascii art. 
    
    Parameters
    ----------
    path_to_image : str
        Path to the image to be converted.
    components : int
        Number of components/characters to convert image into.
    resize : (int, int) 
        Resizing the image. Each pixel will be an ascii character. (None, None) 
        retains default size. A number and None will retain the original ratio.
    ascii_str : str
        The ascii characters to use. If string is longer than components, rest
        of string is ignored.
        
    Returns
    -------
    ascii_art : str
        The image converted to ascii art. 
    '''
    
    # resize image
    image = Image.open(path_to_image)
    horig, vorig = image.size
    if resize[0] is not None and resize[1] is None:
        scale = resize[0]/horig
        resize = (resize[0], int(vorig*scale))
    elif resize[0] is None and resize[1] is not None:
        scale = resize[1]/vorig
        resize = (int(horig*scale), resize[1])
    new_image = image.resize(resize)
    new_image.save('_temp_' + path_to_image)
    
    # open and reshape in matplotlib
    img = mpimg.imread('_temp_' + path_to_image)
    os.remove('_temp_' + path_to_image) # remove temp image
    vpx, hpx, rgb = img.shape
    img.shape = (vpx*hpx, rgb)
    
    # normalise
    scaler = StandardScaler().fit(img)
    img_scale = scaler.transform(img)
    
    # fit and apply k means clustering
    kmeans = KMeans(n_clusters=components)
    kmeans.fit(img_scale)
    img_trans = kmeans.predict(img_scale)
    img_trans.shape = (vpx, hpx)
    
    # convert to ascii art
    ascii_art = ''
    for row in img_trans:
        ascii_row = ''
        for i in row:
            ascii_row = ascii_row + ascii_str[i]
        ascii_art = ascii_art + ascii_row + '\n'
    
    return ascii_art[:-1] # remove last newline
