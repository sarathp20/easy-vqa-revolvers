# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 08:51:01 2022

@author: HP
"""

import os
# Function to rename multiple files
def main():
   path=r'D:\Img\S7ProjectDataset'
   for filename in os.listdir(path):
      my_dest =filename[3:-4]+ ".png"
      my_source =path+"\\"+ filename
      my_dest = path +"\\"+ my_dest
      # rename() function will
      # rename all the files
      os.rename(my_source, my_dest)
      
# Driver Code
if __name__ == '__main__':
   # Calling main() function
   main()