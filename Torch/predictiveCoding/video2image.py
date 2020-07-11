#convert video data to jpg frames
import imageio
import torchvision.transforms as transforms
from torchvision.utils import save_image
from pathos.multiprocessing import ProcessingPool as P
import os
import scipy
import scipy.misc


#Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

def convert_v2imag(inpath,outpath):
    files=os.listdir(inpath)
    for f in files:
        
        if not os.path.isdir(outpath):
            os.mkdir(outpath)
        
        outfolder=outpath+'/'+f.split('.')[0]
        os.chdir(outpath)
        
        os.mkdir(outfolder)
        file=imageio.get_reader(inpath+'/'+f, "ffmpeg")
        print(f)
        for i in range(len(list(file))):
            im_array=list(file)[i]
            scipy.misc.imsave(outfolder+'/'+'frame'+str(i)+'.jpg', im_array)

inpath_header='/Users/lekang/anaconda/tests/Review/Torch/predictiveCoding/UCF-101/'
outpath_header='/Users/lekang/anaconda/tests/Review/Torch/predictiveCoding/ucf101-jpg/'

def convert_folder(inpath_h,outpath_h,threads=6):
    
    p=P(threads)
    
    folders=os.listdir(inpath_h)
    
    p.map(lambda x:convert_v2imag(inpath_h+'/'+x,outpath_h+'/'+x),folders)
    
    


convert_folder(inpath_h='/Users/lekang/anaconda/tests/Review/Torch/predictiveCoding/UCF-101/',
               outpath_h='/Users/lekang/anaconda/tests/Review/Torch/predictiveCoding/ucf101-jpg/',threads=12)
        
