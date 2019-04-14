# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 10:56:54 2019

@author: Victor Zuanazzi
"""

def print_flags(FLAGS):
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))