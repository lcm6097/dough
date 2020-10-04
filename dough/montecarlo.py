# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 00:17:19 2020
MONTE CARLO
@author: melul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import cached_property, partial
from base import DataManipulation