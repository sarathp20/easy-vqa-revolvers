# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 09:03:30 2022

@author: HP
"""

import PIL
import os
import os.path
from PIL import Image

f = r'D:\Img\S7ProjectDataset'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((250,250))
    img.save(f_img)