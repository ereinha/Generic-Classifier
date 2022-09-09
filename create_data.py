import numpy as np
from sklearn.model_selection import train_test_split

signal = np.random.normal(loc=1, scale=.4, size=(100000,3))
background = np.random.normal(loc=0, scale=.2, size=(100000,3))

signal_train, \
signal_valid, \
background_train, \
background_valid = train_test_split(signal, 
                                  background, 
                                  test_size=.25, 
                                  random_state=12)

np.save('./signal_train.npy', signal_train)
np.save('./signal_valid.npy', signal_valid)
np.save('./bg_train.npy', background_train)
np.save('./bg_valid.npy', background_valid)