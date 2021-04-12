#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchvision
import matplotlib.pyplot as plt
plt.show()


# In[2]:


resnet = torchvision.models.resnet18()


# In[3]:


print(resnet)


# In[ ]:


import os
import glob
import pandas as pd
import numpy as np
from skimage import io
from ddks.data.openimages_dataset import LS

dataset = LS()
x, y = next(dataset)
print(x, y)


# In[ ]:




