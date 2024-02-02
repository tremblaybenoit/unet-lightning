import os
import glob

from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class DirDataset(Dataset):
    def __init__(self, img_dir, mask_dir, nx_max=600, ny_max=600, nb_files=50, scale=1, 
                 transforms_stack=None, transforms_img=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.nx = nx_max
        self.ny = ny_max
        self.nb_files = nb_files
        self.transforms_stack = transforms_stack
        self.transforms_img = transforms_img

        try:
            self.ids = [s.split('_.')[0] for s in os.listdir(self.img_dir)]
        except FileNotFoundError:
            self.ids = []

    def __len__(self):
        return 50 #len(self.ids)

    def preprocess(self, img, mask=False):
        w, h = img.size
        _h = int(h * self.scale)
        _w = int(w * self.scale)
        assert _w > 0
        assert _h > 0
        # print('Image size 1: ', img.size, _h, _w)

        # _img = img.resize((_w, _h))
        _img = img.resize((self.nx, self.ny))
        # print('Image size 2: ', img.size, _h, _w)
        _img = np.array(_img)
        # print('Image size before: ', _img.shape)
        if _img.max() > 1:
            _img = _img / 255.
            if mask == True:
                _img = 1.0-_img

        '''
        if _w < self.nx:
           _img = np.pad(_img, ((0, 0), (0, self.nx-w)), mode='constant', constant_values=0)
           for i in range(w, self.nx):
               _img[:, i] = _img[:, w-1]
        #else:
        if _w > self.nx:
            _img = _img[:, 0:self.nx]

        if _h < self.ny:
            _img = np.pad(_img, ((0, self.ny-h), (0, 0)), mode='constant', constant_values=0)
            for i in range(h, self.ny):
                _img[i, :] = _img[h-1, :]
        #else:
        if _h > self.ny:
            _img = _img[0:self.ny, :]
        '''
        
        # print('Image size after: ', _img.shape)
        # print('Image size: ', _img.shape)
        if len(_img.shape) == 2:  ## gray/mask images
            _img = np.expand_dims(_img, axis=-1)
        # print('Image size: ', _img.shape)

        # hwc to chw
        _img = _img.transpose((2, 0, 1))
        # print('Image size: ', _img.shape)

        #if mask == False:
        #    _img = np.concatenate((_img, np.sqrt(_img), np.square(_img)))
        #     _img = np.concatenate((_img, np.sqrt(_img), np.log((_img/255+1.)*5)))

        # _img = np.expand_dims(_img, axis=-1)
        # print('Image size: ', _img.shape)
        # _img[..., 1] = np.sqrt(_img[..., 0])
        # _img[..., 2] = _img[..., 0]**2        

        return _img.astype(np.float32)

    def __getitem__(self, i):
        idx = self.ids[i]
        img_path = sorted(glob.glob(os.path.join(self.img_dir, '*_img.*')))[0:self.nb_files]
        mask_path = sorted(glob.glob(os.path.join(self.mask_dir, '*_mask_.*')))[0:self.nb_files]
        # img_path = sorted(glob.glob(os.path.join(self.img_dir, '*_img.*')))[0:self.nb_files]
        # mask_path = sorted(glob.glob(os.path.join(self.mask_dir, '*_msk.*')))[0:self.nb_files]
        img_files = sorted(glob.glob(img_path[i]))
        mask_files = sorted(glob.glob(mask_path[i]))
        # print(self.ids)

        assert len(img_files) == 1, f'{idx}: {img_files}'
        assert len(mask_files) == 1, f'{idx}: {mask_files}'

        # use Pillow's Image to read .gif mask
        # https://answers.opencv.org/question/185929/how-to-read-gif-in-python/
        img = Image.open(img_files[0])
        img = torch.as_tensor(self.preprocess(img), dtype=torch.float32)
        mask = Image.open(mask_files[0])
        mask = torch.as_tensor(self.preprocess(mask, mask=True), dtype=torch.float32)
        # assert img.size == mask.size, f'{img.shape} # {mask.shape}'

        # with Image.open(img_files[0]) as tmp:
        #    img = torch.as_tensor(self.preprocess(tmp), dtype=torch.float32)
        # with Image.open(mask_files[0]) as tmp:
        #     mask = torch.as_tensor(self.preprocess(tmp, mask=True), dtype=torch.float32)

        if self.transforms_stack is not None:
            stacked = torch.cat([img, mask], dim=0)
            stacked = self.transforms_stack(stacked)
            img, mask = torch.chunk(stacked, chunks=2, dim=0)

        if self.transforms_img is not None:
            img = self.transforms_img(img)

        # mask.append(np.fliplr(mask))
        # mask.append(np.flipud(mask))
        # final_train_data.append(np.rot90(train_x[i], k=1))
        # img.append(np.fliplr(img))
        # img.append(np.flipud(img))
        # img.append(rotate(img, angle=90, mode = 'wrap'))

        #return torch.from_numpy(img).float(), \
        #    torch.from_numpy(mask).float()
        return img, mask
