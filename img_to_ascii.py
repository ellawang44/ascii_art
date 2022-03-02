import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ImageManip:
    '''Manipulate images.
    '''

    def __init__(self, path_to_image, resize=(None, None)):
        '''
        Parameters
        ----------
        path_to_image : str
            Path to the image to be converted.
        resize : (int, int)
            Resizing the image. Each pixel will be an ascii character. (None, None)
            retains default size. A number and None will retain the original ratio.
        '''

        self.path_to_image = path_to_image
        self.resize = resize

    def resize_img(self):
        '''Resize image.
        '''
        
        # resize
        image = Image.open(self.path_to_image)
        horig, vorig = image.size
        if self.resize[0] is not None and self.resize[1] is None:
            scale = self.resize[0]/horig
            self.resize = (self.resize[0], int(vorig*scale))
        elif self.resize[0] is None and self.resize[1] is not None:
            scale = self.resize[1]/vorig
            self.resize = (int(horig*scale), self.resize[1])
        if self.resize[0] is not None and self.resize[1] is not None:
            new_image = image.resize(self.resize)
        else:
            new_image = image
        new_image.save('_temp_' + self.path_to_image)

    def fit(self, components, seed):
        '''Fit the image with k means clustering.
        
        Parameters
        ----------
        components : int
            Number of components/characters to convert image into.
        seed : int
            The random seed to use for KMeans clustering. Can be None.

        Returns
        -------
        data : dict
            img (2darray of ints) is the fitted image, each value is a component. 
            img_orig (2darray of floats) is the original img.
            centers (2darray of floats) is the center of the clusters.
        '''

        # open and reshape in matplotlib
        img = mpimg.imread('_temp_' + self.path_to_image)
        os.remove('_temp_' + self.path_to_image) # remove temp image
        vpx, hpx, rgb = img.shape
        img.shape = (vpx*hpx, rgb)

        # normalise
        scaler = StandardScaler().fit(img)
        img_scale = scaler.transform(img)
        
        # fit and apply k means clustering
        kmeans = KMeans(n_clusters=components, random_state=seed)
        kmeans.fit(img_scale)
        # this seems to keep ordering correct as opposed to predict
        # important for cartoon filter, not so much ascii art
        img_trans = kmeans.labels_ 
        img_trans.shape = (vpx, hpx)

        return {'img':img_trans, 'img_orig':img.reshape(vpx, hpx, rgb),
                'centers':scaler.inverse_transform(kmeans.cluster_centers_)
               }

def img_to_ascii(
    path_to_image, components=20, resize=(None, None),
    ascii_str='-:`!@#$%^&*0123456789qwertyuiopasdfghjklzxcvbnm',
    seed=None):
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
    
    if components > len(ascii_str):
        raise ValueError(
            'ascii_str is not long enough for the number of components specified.'
        )
    
    img_obj = ImageManip(path_to_image=path_to_image, resize=resize)
    img_obj.resize_img()
    img = img_obj.fit(components, seed=seed)['img']

    # convert to ascii art
    ascii_art = ''
    for row in img:
        ascii_row = ''
        for i in row:
            ascii_row = ascii_row + ascii_str[i]
        ascii_art = ascii_art + ascii_row + '\n'
    
    return ascii_art[:-1] # remove last newline

def cartoon_filter(path_to_image, out_image, components=10, resize=(None, None),
        colours=None, seed=None):
    '''Filter the image through a cartoon effect.
    '''

    if colours is not None:
        if len(colours) != len(components):
            raise ValueError(
                    'Given colours are not the same length as components.'
            )
    
    img_obj = ImageManip(path_to_image=path_to_image, resize=resize)
    img_obj.resize_img()
    data = img_obj.fit(components, seed=seed)
    
    # place all the rgb colours into the correct places
    img = data['img']
    centers = data['centers']
    filt_img = []
    for i, row in enumerate(img):
        filt_img_row = []
        for j, pix in enumerate(row):
            filt_img_row.append(centers[pix])
        filt_img.append(filt_img_row)
    filt_img = np.array(filt_img)

    # save image
    plt.imsave(out_image, filt_img)
