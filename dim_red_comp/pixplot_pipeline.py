import os
import pandas as pd
import numpy as np
import sys
import time
import re
import pickle


class ProgressBar():
    def __init__(self,N,BarCounts = 40,newline_at_the_end=True, 
                    ProgressCharacter = "*", UnProgressCharacter = ' ', step = 1, dynamic_step = True):
        '''
        BarCounts : total number of bars in the progress bar
        newline_at_the_end : insert a newline when the loop ends
        ProgressCharacter : character to use for the progress bar (default is a full block)
        UnProgressCharacter : character to use for the remainder of progress bar (default is space)
        step : skip this many counts before updating the progress bar
        '''
        self.Time0 = time.time()
        self.BarCounts = BarCounts
        self.N = N
        self.i = 0
        self.newline_at_the_end = newline_at_the_end
        self.ProgressCharacter = ProgressCharacter
        self.UnProgressCharacter = UnProgressCharacter
        self.step = step    
        self.PrevWriteInputLength = 0
        self.dynamic_step = dynamic_step
        
    def Update(self,Text = '',no_variable_change = False, PrefixText=''):
        '''
        Text L: text to show during the update
        no_variable_change : do not update the internal counter if this is set to True
        '''        
        if not no_variable_change:
            self.i = self.i + 1
            
        if (self.i % self.step == 0) | (self.i==self.N):
            CurrentTime = (time.time()-self.Time0)/60.0
            CurrProgressBarCounts = int(self.i*self.BarCounts/self.N)
            self.WriteInput = u'\r%s|%s%s| %.1f%% - %.1f / %.1f minutes - %s'%(
                                                          PrefixText,
                                                          self.ProgressCharacter*CurrProgressBarCounts, 
                                                          self.UnProgressCharacter*(self.BarCounts-CurrProgressBarCounts), 
                                                          100.0*self.i/self.N, 
                                                          CurrentTime, 
                                                          CurrentTime*self.N/self.i,
                                                          Text)
            ExtraSpaces = ' '*(self.PrevWriteInputLength - len(self.WriteInput))    # Needed to clean remainder of previous text
            self.PrevWriteInputLength = len(self.WriteInput)                                        
            sys.stdout.write(self.WriteInput+ExtraSpaces)
            sys.stdout.flush()
            if (not no_variable_change) & self.newline_at_the_end & (self.i==self.N):
                print('\n')
    def NestedUpdate(self,Text = '',ProgressBarObj=None,no_variable_change = False, PrefixText=''): # nest this progressbar within another progressbar loop
        if ProgressBarObj!=None:
            ProgressBarObj.newline_at_the_end = False
            # assert not ProgressBarObj.newline_at_the_end, 'The object prints newline at the end. Please disable it.'
            self.newline_at_the_end = False
            no_variable_change = False
            PrefixText=ProgressBarObj.WriteInput
        self.Update(Text = Text,no_variable_change = no_variable_change, PrefixText=PrefixText)

def main_tiling(**config):
    print("The tiling procedure has started.") 
    for images_path in config['slide']:
        isFile = os.path.isfile(images_path)
        isDirectory = os.path.isdir(images_path)

        tiles_path = config['tile_dir']
        mkdir_if_not_exist(tiles_path)
        tile_size = config['tile_size'] 

        if isFile == True:
            print("The slide: " + images_path + " will be tiled")
            t0 = time.time()
            try:
                TileSVS_and_crop_using_deepzoom(images_path, tiles_path, tile_size)
            except:
                savedir = config['pyramid_dir']
                pyramid_im = make_pyramid(images_path, savedir, tile_size)
                TileSVS_and_crop_using_deepzoom(pyramid_im, tiles_path, tile_size)
            tf = time.time() - t0
            print(f'Tiling time: {tf:.2f}s')

        elif isDirectory == True:
            print("All the slides from the directory: " + images_path + " will be tiled")
            for file in os.listdir(images_path):
                t0 = time.time()
                path = os.path.join(images_path, file)
                try:
                    TileSVS_and_crop_using_deepzoom(path, tiles_path, tile_size)
                except:
                    savedir = config['pyramid_dir']
                    pyramid_im = make_pyramid(path, savedir, tile_size)
                    TileSVS_and_crop_using_deepzoom(pyramid_im, tiles_path, tile_size)
                tf = time.time() - t0
                print(f'Tiling time: {tf:.2f}s')


def main_model(**config):
    config = check_model(**config)
    
    model_name = config['model_name']
    tiles_path = config['tile_dir']
    results_path = config['latent_dir']
    processing_unit = config['processing_unit']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    tile_resize = config['tile_resize']
    tile_size = config['tile_size']

    if config['no_valid_model'] == False:
        print('\n')
        print("Latent space comptation has started.")

        isFile = os.path.isfile(tiles_path)
        isDirectory = os.path.isdir(tiles_path)

        if isFile == True:
            t0 = time.time()
            data = apply_model(model_name, tiles_path, results_path, processing_unit, batch_size, num_workers, tile_resize, tile_size)
            tf = time.time() - t0
            print(f'Inference time: {tf:.2f}s')

        elif isDirectory == True:
            for file in os.listdir(tiles_path):
                t0 = time.time()
                path = os.path.join(tiles_path, file)
                data = apply_model(model_name, path, results_path, processing_unit, batch_size, num_workers, tile_resize, tile_size)
                tf = time.time() - t0
                print(f'Inference time: {tf:.2f}s')

    return config

def check_model(**config):
    model_name = ''
    if config['timm_model'] != None or config['model'] != None:
        if config['timm_model'] != None and config['model'] == None:
            model_name = config['timm_model']
            dic={'model_name': model_name}
        elif config['model'] != None and config['timm_model'] == None :
            model_name = config['model']
            dic={'model_name': model_name}
        elif config['model'] != None and config['timm_model'] != None :
            print('Two models specified, only one permitted')
            dic={'no_valid_model': True}
    else:
        print('no valid model specified')
        dic={'no_valid_model': True}
    config.update(dic)
    return config
        
def main_pixplot_input(**config):
    config = check_model(**config)
    if config['no_valid_model'] == False:
        print("The pixplot input file is being generated.")  
        t0 = time.time()
        config = pixplot(**config)
        tf = time.time() - t0
        print(f'Pixplot input file generation time: {tf:.2f}s')
    return config

def make_pyramid(path, savedir, tile_size):
    import tifffile
    import cv2
    image = tifffile.imread(path, key=0)
    h, w, s = image.shape
    slide_name = os.path.basename(path)
    mkdir_if_not_exist(savedir)
    save_path = os.path.join(savedir, slide_name)
    with tifffile.TiffWriter(save_path, bigtiff=True) as tif:
        level = 0
        while True:
            tif.save(
                image,
                software='Glencoe/Faas pyramid',
                metadata=None,
                tile=(tile_size, tile_size),
                resolution=(1000/2**level, 1000/2**level, 'CENTIMETER'),
                # compress=1,  # low level deflate
                # compress=('jpeg', 95),  # requires imagecodecs
                # subfiletype=1 if level else 0,
            )
            if max(w, h) < 256:
                break
            level += 1
            w //= 2
            h //= 2
            image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    print(f"A pyramid version of the input image has been made in {savedir}.") 
    return save_path

def TileSVS_and_crop_using_deepzoom(File, outdir, tile_size, SaveToFile=True,  tile_level_step_down=1,
                                    bg_th=220, max_bg_frac=0.5, ProgressBarObj=None, no_cropping=False):
    
    import openslide as ops
    from openslide import deepzoom
    
    Slide = ops.OpenSlide(File) 

    pixel_size = 'Unknown'
    if "openslide.mpp-x" in Slide.properties:
        pixel_size = str(round(float(Slide.properties['openslide.mpp-x'][0:6]),2))
    elif 'tiff.ImageDescription' in Slide.properties and pixel_size == 'Unknown_pixel_size':
        pixel_size = str(round(10000/float(Slide.properties['tiff.XResolution']),2))
    
    slide_name = os.path.basename(File)
    outdir = outdir + "/" + pixel_size + "_" + slide_name

    tiles = ops.deepzoom.DeepZoomGenerator(Slide, tile_size=tile_size, overlap=0, limit_bounds=False)
    
    assert (np.round(np.array(tiles.level_dimensions[-1]) / np.array(tiles.level_dimensions[-2])) == 2).all(), \
        'level_dimension[-2] should be almost twice smaller than level_dimension[-1] for proper conversion between 20x<->40x'
    
    mkdir_if_not_exist(outdir) 
    
    """if len(os.listdir(outdir)) != 0:
        return False"""
    SaveToFolder = outdir + '/preprocessing/'
    mkdir_if_not_exist(SaveToFolder)
    tilesdir = outdir + '/processing/tiles/' #/processing added
    mkdir_if_not_exist(tilesdir)

    tile_level = tiles.level_count - tile_level_step_down    
    tile_side = np.array(tiles.get_tile(tile_level, (0, 0))).shape

    assert tile_side[0] == tile_side[1]
    tile_side = tile_side[0]
    tiles_size = tiles.level_tiles[tile_level]
    
    coordratios = {'h': np.array([0., 1.]), 'v': np.array([0., 1.])}
    xh = (coordratios['h'] * tiles_size[0]).astype(int)
    xv = (coordratios['v'] * tiles_size[1]).astype(int)
    
    Xdims = (xh[1] - xh[0], xv[1] - xv[0])
    num_tiles = np.prod(Xdims)
    Pbar = ProgressBar(num_tiles, step=1)
    
    general_path = SaveToFolder + '/general.csv'
    
    fi = open(general_path, 'w')
    for prop in Slide.properties:
        fi.write(str(prop) + "," + str(Slide.properties[prop]))
        fi.write('\n') 
               
    fi.write("slide_dimension_0" + "," + str(Slide.dimensions[0]))
    fi.write('\n')
    fi.write("slide_dimension_1" + "," + str(Slide.dimensions[1]))
    fi.write('\n')
    fi.write("tile_size" + "," + str(tile_size))
    fi.write('\n')
    fi.write("number_of_tiles_in_original_slide_0" + "," + str(tiles_size[0]))
    fi.write('\n')
    fi.write("number_of_tiles_in_original_slide_1" + "," + str(tiles_size[1]))
    fi.write('\n')
    fi.write("first_tile_0" + "," + str(xh[0]))
    fi.write('\n')
    fi.write("last_tile_0" + "," + str(xh[1]))
    fi.write('\n')
    fi.write("first_tile_1" + "," + str(xv[0]))
    fi.write('\n')
    fi.write("last_tile_1" + "," + str(xv[1]))
    fi.write('\n')
    fi.write("number_of_tiles_in_preprocessed_slide[0]" + "," + str(Xdims[0]))
    fi.write('\n')
    fi.write("number_of_tiles_in_preprocessed_slide[1]" + "," + str(Xdims[1]))
    fi.write('\n')
    fi.close()
    
    
    coordinates_path = SaveToFolder + '/coordinates.csv'
    fi = open(coordinates_path, 'w')    
    
    segmentation = CropSlideCoordRatiosV4(Slide, (tiles_size[0],tiles_size[1]), min_distance='infer', 
                                                SaveToFolder = SaveToFolder, ShowIntermediateStates = False)

    
    x_range = range(Xdims[0])
    y_range = range(Xdims[1])
    
    

    if segmentation.shape[1] - 1 < max(range(Xdims[0])) + 1:
        x_range = range(segmentation.shape[1])
    if segmentation.shape[0] - 1 < max(range(Xdims[1])) + 1:
        y_range = range(segmentation.shape[0])


    for m in x_range:
        for n in y_range:
            Pbar.NestedUpdate('(%d/%d,%d/%d)' % (m, Xdims[0], n, Xdims[1]), ProgressBarObj=ProgressBarObj)
            tile = tiles.get_tile(tile_level, (m + xh[0], n + xv[0]))
            tile_coordinates = tiles.get_tile_coordinates(tile_level, (m + xh[0], n + xv[0]))
            if SaveToFile:
                if segmentation[n,m] == True:
                    tile_np = np.array(tile)
                    if tile_np.shape[0] == tile_size and tile_np.shape[1] == tile_size:
                        #print(tile_np.shape[1])
                        #print(tile_np.shape[0])
                        outfile = tilesdir + (str(slide_name) + '_' + 'tile_%d_%d.png' % (m, n))
                        tile.save(outfile, "PNG", quality=100)
                        fi.write(str(slide_name) + '_' + 'tile_%d_%d.png' % (m, n) + ",")
                        fi.write(str(m) + ",")
                        fi.write(str(n) + ",")
                        fi.write(str(tile_coordinates[0][0]) + ",")
                        fi.write(str(tile_coordinates[0][1]) + ",")
                        fi.write('\n')
            else:
                tile = np.array(tile)
                X[m, n, :tile.shape[0], :tile.shape[1], :] = tile
    fi.close()
    Slide.close()

def CropSlideCoordRatiosV4(Slide, Slide_size,
                           min_distance, SaveToFolder,
                           ShowIntermediateStates):  
    
    import matplotlib as mpl
    import matplotlib.pyplot as pl
    import skimage.color as skc
    from skimage.filters import threshold_otsu
    
    assert (SaveToFolder == None) or (type(SaveToFolder) == str), 'Proper save folder not provided.'
    
    if ShowIntermediateStates or (SaveToFolder != None):
        if ShowIntermediateStates:
            mpl.use('Agg')
            #pl.use('Agg')
        else:
            mpl.use('pdf')
            #pl.use('pdf')

    thumb = np.asarray(
        Slide.get_thumbnail(Slide_size))  # create thumbnail (benefits: smaller, less computation and memory)

    imgray = 1 - skc.rgb2gray(thumb)  # convert to gray scale

    segmentation = imgray > threshold_otsu(imgray)  # threshold according to otsu's method
        
    if SaveToFolder != None:
        mkdir_if_not_exist(SaveToFolder)
            
        pl.figure()
        pl.axis('off')
        pl.imshow(thumb);
        pl.savefig(os.path.join(SaveToFolder, 'slide.png'), dpi=1000, bbox_inches='tight', pad_inches=0)
        pl.figure()
        pl.axis('off')
        pl.imshow(segmentation);
        pl.savefig(os.path.join(SaveToFolder, 'binarized.png'), dpi=1000, bbox_inches='tight', pad_inches=0)
                

    return segmentation
    
def apply_model(model_name, tiles_path, results_path, processing_unit, batch_size, num_workers, tile_resize, tile_size):   
    #resnet
    #import requests #?
    #import json #?
    #import openslide  #to clean
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import ToTensor
    from torchvision import transforms

    torch.cuda.empty_cache()
    #t0 = time.time()
    
    image_name_full = os.path.basename(tiles_path)
    #print(image_name_full)
    #image_name = re.split('_',image_name_full)[1]
    pixel_size = re.split('_',image_name_full)[0]
    image_name = image_name_full.replace(pixel_size+"_","")
    print(image_name)
    #print(os.path.isfile(model_name))
    if os.path.isfile(model_name)==False:
        import timm
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        if processing_unit != "cpu":
            print("GPU are used")
            cuda_unit = torch.device(processing_unit) 
            model = model.cuda(cuda_unit)
    
    elif os.path.isfile(model_name)==True:
        model = torchvision.models.__dict__['resnet18'](pretrained=False)
        model_path = config['model']
        model_name = os.path.basename(model_path)
        if processing_unit != "cpu":
            print("GPU are used")
            #change1
            cuda_unit = torch.device(processing_unit) 
            state = torch.load(model_path, map_location=processing_unit)
            state_dict = state['state_dict']        
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
            model = load_model_weights(model, state_dict)
            model.fc = torch.nn.Sequential()
            #change2
            model = model.cuda(cuda_unit)
        else:
            print("CPU are used")
            state = torch.load(model_path, map_location=torch.device(processing_unit))
            state_dict = state['state_dict']
            for key in list(state_dict.keys()):
                state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
            model = load_model_weights(model, state_dict)
            model.fc = torch.nn.Sequential()

    class ImageFolderWithPaths(datasets.ImageFolder):
        # override the __getitem__ method. this is the method that dataloader calls
        def __getitem__(self, index):
            # this is what ImageFolder normally returns 
            original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
            # the image file path is added
            path = self.imgs[index][0]
            # make a new tuple that includes original and the path
            tuple_with_path = (original_tuple + (path,))
            return tuple_with_path
    if tile_size == tile_resize:
        dataset = ImageFolderWithPaths(tiles_path + "/processing", transform=ToTensor())
    else:
        dataset = ImageFolderWithPaths(tiles_path + "/processing", transform=transforms.Compose([transforms.Resize(tile_resize), transforms.ToTensor()]))
    
    dataload = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    file_path = results_path + "/" + pixel_size + "/" + model_name + "/" + image_name + "/"
    mkdir_if_not_exist(file_path)

    for i, (images, labels, paths) in enumerate(dataload):
#        print(i)
        if processing_unit != "cpu":
            images = images.cuda(cuda_unit)
        repLat = model(images)
        array = repLat.detach().cpu().numpy()
        torch.cuda.empty_cache()
        #print(array)
        df = pd.DataFrame(array)
        df.insert(0, "tile_name", paths[0:(batch_size+1)])
        df.insert(1, "image_name", image_name)
        if i == 0:
            data = df
        else:
            data = pd.concat([data,df])
    print("latent_space_dimension: ", data.shape)
    data.index = data['tile_name']
    data = data.drop(['tile_name'], axis = 1)

    data.to_csv(file_path + "_repLat" + model_name + ".csv", header=False)
    #tf = time.time() - t0
    #print(f'inference time: {tf:.2f}s')
    return data

def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir

"""def model_name_from_latent_dir(**config):
    latent_space_folder = config['latent_dir']
    count = 0
    for folder in os.listdir(latent_space_folder):
        print(folder)
        folder = os.path.join(latent_space_folder, folder)
        for model_folder in os.listdir(folder):
            print(model_folder)
            count += 1
    if count == 1:
        return model_folder
    else:
        print('no valid model specified')"""
            

def check_model_pixplot(**config):
    model_name = ''
    if config['timm_model'] != None or config['model'] != None:
        if config['timm_model'] != None and config['model'] == None:
            model_name = config['timm_model']
            dic={'model_name': model_name}
        elif config['model'] != None and config['timm_model'] == None :
            model_path = config['model']
            model_name = os.path.basename(model_path)
            dic={'model_name': model_name}
        elif config['model'] != None and config['timm_model'] != None :
            print('Two models specified, only one permitted')
            dic={'no_valid_model': True}
    else:
        print('no valid model specified')
        dic={'no_valid_model': True}
    config.update(dic)
    return config

def pixplot(**config):
    config = check_model_pixplot(**config)
    model_name = config['model_name']
    all_pixel_slides = []
    all_slide_names = []
    all_pixel_sizes = []
    all_pixel_input_files = []
    all_latent_space_files = []
    all_coordinates_files = []
    all_tiles_metadata_files = []
    all_slide_thumbnail_files =[]
    all_pixplot_input_files = []
    
    for file in os.listdir(config['tile_dir']):
        path = os.path.join(config['tile_dir'], file)
        pixel_slide = os.path.basename(path)
        #slide_name = re.split('_',pixel_slide)[1]
        pixel_size = re.split('_',pixel_slide)[0]
        slide_name = pixel_slide.replace(pixel_size+"_","")
        print(slide_name)
        latent_suffix = pixel_size + '/' + model_name + '/' + slide_name + '/' + '_repLat' + model_name + '.csv'
        latent_space_file = os.path.join(config['latent_dir'], latent_suffix)
        coordinates_suffix = pixel_slide + '/preprocessing/coordinates.csv'
        coordinates_file = os.path.join(config['tile_dir'], coordinates_suffix)
        #general_suffix = pixel_slide + '/preprocessing/general.csv'
        #tiles_metadata_file = os.path.join(config['tile_dir'], general_suffix)
        slide_thumbnail_suffix = pixel_slide + '/preprocessing/slide.png'
        slide_thumbnail_file = os.path.join(config['tile_dir'], slide_thumbnail_suffix)
        pixplot_input_folder_suffix = pixel_size + '/' + model_name + '/' + slide_name 
        pixplot_input_file_suffix = pixplot_input_folder_suffix + '/' + 'pixplot_metadata.csv' 
        pixplot_input_folder = os.path.join(config['pixplot_in_dir'], pixplot_input_folder_suffix)
        #print(pixplot_input_folder)
        mkdir_if_not_exist(pixplot_input_folder)
        pixplot_input_file = os.path.join(pixplot_input_folder, 'pixplot_metadata.csv')
        all_pixel_slides.append(pixel_slide)
        all_slide_names.append(slide_name)
        all_pixel_sizes.append(pixel_size)
        all_pixel_input_files.append(pixplot_input_file)
        all_latent_space_files.append(latent_space_file)
        all_coordinates_files.append(coordinates_file)
        #all_tiles_metadata_files.append(tiles_metadata_file)
        all_slide_thumbnail_files.append(slide_thumbnail_file)
        all_pixplot_input_files.append(pixplot_input_file)
        
        """dic = {
        'all_pixel_slides': all_pixel_slides,
        'all_slide_names': all_slide_names,
        'all_pixel_sizes': all_pixel_sizes,
        'all_pixel_input_files': all_pixel_input_files,
        'all_latent_space_files': all_latent_space_files,
        'all_coordinates_files': all_coordinates_files,
        'all_tiles_metadata_files': all_tiles_metadata_files,
        'all_slide_thumbnail_files': all_slide_thumbnail_files,
        'all_pixplot_input_files': all_pixplot_input_files
        }"""
        dic = {
        'all_pixel_slides': all_pixel_slides,
        'all_slide_names': all_slide_names,
        'all_pixel_sizes': all_pixel_sizes,
        'all_pixel_input_files': all_pixel_input_files,
        'all_latent_space_files': all_latent_space_files,
        'all_coordinates_files': all_coordinates_files,
        'all_slide_thumbnail_files': all_slide_thumbnail_files,
        'all_pixplot_input_files': all_pixplot_input_files
        }
        
        config.update(dic)
        pixplot_input_generation(**config)
    return config


def pixplot_input_generation(**config):
    import umap.umap_ as umap
    count = len(config['all_slide_names'])
    for i in range(count):
        latent_space_file = config['all_latent_space_files'][i]
        coordinates_file = config['all_coordinates_files'][i]
        #tiles_metadata_file = config['all_tiles_metadata_files'][i]
        pixplot_input_file = config['all_pixplot_input_files'][i]
        latent = pd.read_csv(latent_space_file, header=None, index_col=0)
        latent = latent.drop(columns=[1])
        latent = latent.sort_index()
        #print(latent.shape)

        coordinates = pd.read_csv(coordinates_file, header=None, index_col=0)
        coordinates = coordinates.drop(columns=[5])
        coordinates.columns = ['x.tile', 'y.tile', 'x.pixel', 'y.pixel']
        coordinates = coordinates.sort_index()
        #print(coordinates.shape)

        #tiles_meta_data = readcsv(tiles_metadata_file)

        # Make and plot UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(latent)
        #print(embedding.shape)

        pixplot_data = pd.DataFrame({'filename': latent.index,
                                     'lat' : 1 + max(coordinates['y.tile']) - coordinates['y.tile'],
                                     'lng' : coordinates['x.tile'],
                                    'umap_x' : embedding[:, 0], 'umap_y' : embedding[:, 1]})

        pixplot_data.to_csv(pixplot_input_file, index=False)

def readcsv(file):
    df=pd.read_csv(file, header=None, sep='\n')
    df['attribute'] = pd.DataFrame(list(df[0].str.split(',')))[0].values
    df['value'] = pd.DataFrame(list(df[0].str.split(',')))[1].values
    df.index=df['attribute']
    df=df.drop(columns=[0, 'attribute'])
    df=df.T
    return df 


def process_images(**config):
    if config['tiling']==True:
        main_tiling(**config)
    if config['inference']==True:
        config = main_model(**config)
    if config['pixplot']==True:
        config = main_pixplot_input(**config)
    return config
        

if __name__ == '__main__':
    current_folder = os.getcwd()
    config = pickle.load(open(os.path.join(current_folder, 'config.txt'), "rb"))
    config = process_images(**config)
    pickle.dump(config, open(os.path.join(current_folder, 'config.txt'), "wb"))
        



        
        