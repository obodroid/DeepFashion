#!/usr/bin/env python
# coding: utf-8

# In[7]:


import shutil
import os
import re
import cv2
# will use them for creating custom directory iterator
import numpy as np
from six.moves import range
# regular expression for splitting by whitespace
splitter = re.compile("\s+")


# In[6]:


dataset_path = '/src/dataset/'
base_path = '/src/dataset/process/'
process_path = '/src/dataset/process/'
image_path = '/src/fashion_data/Img/img/'
img_train_path = process_path + 'train/'
img_eval_path = process_path + 'val/'
img_test_path = process_path + 'test/'
anno_path = '/src/fashion_data/Anno/'
eval_path = '/src/fashion_data/Eval/'
log_path = '/src/logs/'
output_path = '/src/output/'


# In[3]:


def process_folders():
    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open('{}list_eval_partition.txt'.format(eval_path), 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_all = [(v[0][4:], v[0].split('/')[1].split('_')[-1], v[1]) for v in list_eval_partition]
    print(list_all[0])
    # Put each image into the relevant folder in train/test/validation folder
    for element in list_all:
        if not os.path.exists(os.path.join(base_path, element[2])):
            os.mkdir(os.path.join(base_path, element[2]))
        if not os.path.exists(os.path.join(os.path.join(base_path, element[2]), element[1])):
            os.mkdir(os.path.join(os.path.join(base_path, element[2]), element[1]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                              element[0].split('/')[0])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1])),
                     element[0].split('/')[0]))
        shutil.move(os.path.join(image_path, element[0]),
                    os.path.join(os.path.join(os.path.join(base_path, element[2]), element[1]), element[0]))


# In[4]:


process_folders()


# In[5]:


def create_dict_bboxes(list_all, split='train'):
    lst = [(line[0], line[1], line[3], line[2]) for line in list_all if line[2] == split]
    lst = [("".join(base_path + line[3] + '/' + line[1] + line[0][3:]), line[1], line[2]) for line in lst]
    lst_shape = [cv2.imread(line[0]).shape for line in lst]
#     print("line round")
    lst = [(line[0], line[1], (round(float(line[2][0]) / shape[1], 2), round(float(line[2][1]) / shape[0], 2), round(float(line[2][2]) / shape[1], 2), round(float(line[2][3]) / shape[0], 2)), shape) for line, shape in zip(lst, lst_shape)]
#     for line in lst:
#         print(line)
    dict_ = {"/".join(line[0].split('/')[5:]): {'x1': line[2][0], 'y1': line[2][1], 'x2': line[2][2], 'y2': line[2][3], 'shape': line[3]} for line in lst}
    return dict_

def get_dict_bboxes():
    with open('{}list_category_img.txt'.format(anno_path), 'r') as category_img_file,             open('{}/list_eval_partition.txt'.format(eval_path), 'r') as eval_partition_file,             open('{}/list_bbox.txt'.format(anno_path), 'r') as bbox_file:
        list_category_img = [line.rstrip('\n') for line in category_img_file][2:]
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_bbox = [line.rstrip('\n') for line in bbox_file][2:]

        list_category_img = [splitter.split(line) for line in list_category_img]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        list_bbox = [splitter.split(line) for line in list_bbox]

        list_all = [(k[0], k[0].split('/')[1].split('_')[-1], v[1], (int(b[1]), int(b[2]), int(b[3]), int(b[4])))
                    for k, v, b in zip(list_category_img, list_eval_partition, list_bbox)]

        list_all.sort(key=lambda x: x[1])
        dict_train = create_dict_bboxes(list_all)
        dict_val = create_dict_bboxes(list_all, split='val')
        dict_test = create_dict_bboxes(list_all, split='test')

        return dict_train, dict_val, dict_test


# In[6]:


testStr = '/src/dataset/process/train/Anorak/Hooded_Cotton_Canvas_Anorak/img_00000004.jpg'
"/".join(testStr.split('/')[5:])


# In[7]:


shape = (300, 258, 3)
bb = (64, 51, 218, 290)
round(float(bb[0]) / float(shape[1]), 2)


# In[120]:


list_all = get_dict_bboxes()


# In[97]:


# line = list_all[0]
# line = (line[0], line[1], line[3], line[2])
# print(line)
# line = ("".join(base_path + line[3] + '/' + line[1] + line[0][3:]), line[1], line[2])
# print(line[0])
# cv2.imread(line[0]).shape


# In[31]:


from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K
from keras.preprocessing import image


# In[9]:


model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')


# In[23]:


for layer in model_resnet.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False


# In[24]:


x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x)
y = Dense(46, activation='softmax', name='img')(x)


# In[25]:


x_bbox = model_resnet.output
x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)


# In[26]:


final_model = Model(inputs=model_resnet.input,
                    outputs=[y, bbox])


# In[27]:


print(final_model.summary())


# In[28]:


opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)


# In[29]:


final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_squared_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mse']})


# In[3]:


train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()


# In[4]:


class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes = None, target_size=(256, 256), color_mode = 'rgb', classes=None, class_mode = 'categorical', batch_size = 32,shuffle = True, seed=None, data_format=None, save_to_dir=None, save_prefix = '', save_format = 'jpeg', follow_links = False):
        super(DirectoryIteratorWithBoundingBoxes, self).__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def next(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['x1'], bounding_box['y1'], bounding_box['x2'], bounding_box['y2']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y


# In[149]:


dict_train, dict_val, dict_test = get_dict_bboxes()


# In[10]:


train_iterator = DirectoryIteratorWithBoundingBoxes(img_train_path, train_datagen,  target_size=(200, 200))


# In[11]:


train_iterator.class_indices


# In[14]:


generator= train_datagen.flow_from_directory(img_train_path)


# In[10]:


label_map = (generator.class_indices)
label_map


# In[9]:


train_iterator = DirectoryIteratorWithBoundingBoxes(img_train_path, train_datagen, bounding_boxes=dict_train, target_size=(200, 200))


# In[151]:


test_iterator = DirectoryIteratorWithBoundingBoxes(img_eval_path, test_datagen, bounding_boxes=dict_val,target_size=(200, 200))


# In[152]:


lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)
tensorboard = TensorBoard(log_dir=log_path)
early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
checkpoint = ModelCheckpoint('{}model.h5'.format(output_path))


# In[141]:


def custom_generator(iterator):    
    while True:
        batch_x, batch_y = iterator.next()        
        yield (batch_x, batch_y)


# In[181]:


steps_per_epoch = 2000 #2000
epochs = 200 #200
validation_steps = 200 #200


# In[182]:


final_model.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=steps_per_epoch,
                          epochs=epochs, validation_data=custom_generator(test_iterator),
                          validation_steps=validation_steps,
                          verbose=2,
                          shuffle=True,
                          use_multiprocessing=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12)


# In[155]:


test_datagen = ImageDataGenerator()


# In[8]:


test_iterator = DirectoryIteratorWithBoundingBoxes(img_test_path, test_datagen, bounding_boxes=dict_test, target_size=(200, 200))


# In[157]:


scores = final_model.evaluate_generator(custom_generator(test_iterator), steps=2000)

print('Multi target loss: ' + str(scores[0]))
print('Image loss: ' + str(scores[1]))
print('Bounding boxes loss: ' + str(scores[2]))
print('Image accuracy: ' + str(scores[3]))
print('Top-5 image accuracy: ' + str(scores[4]))
print('Bounding boxes error: ' + str(scores[5]))


# In[17]:


testImage = img_test_path + 'Blouse/Sheer_Pleated-Front_Blouse/img_00000005.jpg'
img = image.load_img(testImage, target_size=(200, 200))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = final_model.predict(x)
print(preds[0])


# In[12]:


from keras.models import load_model
loaded_model = load_model('{}model_46_1_27032019.h5'.format(output_path))


# In[75]:


testImages = []
testImages.append(img_test_path + 'Blouse/Sheer_Pleated-Front_Blouse/img_00000053.jpg')
testImages.append(img_test_path + "Skirt/Zippered_Faux_Leather_Mini_Skirt/img_00000018.jpg")
testImages.append(img_test_path + "Jeans/Classic_Slim_Denim_Jeans/img_00000045.jpg")
testImages.append("/src/dataset_prediction/images/" + "tshirt.jpg")
testImages.append("/src/dataset_prediction/images/" + "crop1_1.jpg")
testImages.append("/src/dataset_prediction/images/" + "crop2_2.jpg")

testImageDicts = {}

for testImagePath in testImages:
    img = image.load_img(testImagePath, target_size=(200, 200))
    # print(img.width,img.height)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = loaded_model.predict(x)
    predit_class_index = np.argmax(preds[0])
    confidence = preds[0].max()
    predict_bbox = preds[1].reshape(-1)
    # predict_bbox = predict_bbox[0]*
    print(confidence,predict_bbox)
    predict_class_label = train_iterator.class_indices.keys()[predit_class_index]
    print (predit_class_index,class_label)
    testImageDicts[testImagePath] = {"predit_class_index":predit_class_index,"predict_class_label":predict_class_label,"confidence":confidence,"predict_bbox":predict_bbox}


# In[78]:


get_ipython().magic(u'matplotlib inline')
#The line above is necesary to show Matplotlib's plots inside a Jupyter Notebook
from matplotlib import pyplot as plt

#Import image
for testImagePath in testImages:
    pltImage = cv2.imread(testImagePath)
    shape = pltImage.shape
    print(shape)
    print testImageDicts[testImagePath]
    predict_bbox = testImageDicts[testImagePath]["predict_bbox"]
    scale_bbox = int(predict_bbox[0]*shape[1]),int(predict_bbox[1]*shape[0]),int(predict_bbox[2]*shape[0]),int(predict_bbox[3]*shape[1])
    print scale_bbox
    pltImage = cv2.rectangle(pltImage,(scale_bbox[0],scale_bbox[1]),(scale_bbox[2],scale_bbox[3]),(0,255,0),3)
    #Show the image with matplotlib
    plt.imshow(pltImage)
    plt.show()


# In[9]:


index = np.argmax(preds[0])
print(index)
print label_map.keys()[label_map.values().index(index)]

