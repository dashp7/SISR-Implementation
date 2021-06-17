#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import utils
import models
import tensorflow as tf
#from tensorflow.keras import losses, optimizers, callbacks


# In[2]:


base=os.getcwd()
train_dir=os.path.join(base,"DIV2K_train_HR")
test_dir=os.path.join(base, "DIV2K_test_HR")
print("Base = {}".format(base))
print("Train = {}".format(train_dir))
print("Test = {}".format(test_dir))


# In[3]:


crop_size = 500
sr_factor = 4
input_size = crop_size // sr_factor
batch_size = 24
val_split = 0.1
utils.showplot =0 #Set 1 to show plots in output
utils.res_fraction = 0.6 #Fraction of the original resolution at which the plots will be saved


# In[4]:


ds_train, ds_valid = utils.define_datasets(train_dir, batch_size, crop_size, val_split)
ds_train, ds_valid = utils.map_datasets(ds_train, ds_valid)
test_paths = utils.generate_test_paths(test_dir)
ds_train, ds_valid = utils.transform_ds(ds_train, ds_valid, input_size, sr_factor)


# In[5]:


model_name ="EDSR"
model = models.EDSR(sr_factor=sr_factor, channels=1, res_blocks = 4)
model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20)
callbacks = [utils.SRCallback(test_paths, sr_factor, model_name), early_stop]
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

epochs = 101
model.compile(optimizer=optimizer, loss=loss_fn,)
r = model.fit(ds_train, epochs=epochs, callbacks=callbacks, validation_data=ds_valid, verbose=1)
model.save(model_name + '.h5')


# In[6]:


saved_model = tf.keras.models.load_model(model_name + '.h5')
utils.inference(saved_model, test_paths, sr_factor, model_name)


# In[7]:


utils.training_stats(r, model_name)


# In[ ]:




