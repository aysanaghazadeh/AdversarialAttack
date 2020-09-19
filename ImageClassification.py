import torchattacks
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import decode_predictions as DPResNet50
import glob
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import time
from keras.applications.vgg16 import preprocess_input as piVGG16
from keras.applications.resnet50 import preprocess_input as piResNet50
from pickle import dump
from keras import losses


directory = '7415-10564-bundle-archive/**/**/*.png'
filename_image = 'images.npy'
filename_label = 'labels.npy'

def read_files(directory):
    img_infos = []
    c = 0
    files = glob.glob(directory)
    time0 = time.time()
    for fp in files:
        image = load_img(fp, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        if c == 0:
            images = image
        else:
            images = np.concatenate((images, image))
        img_infos.append(int(str(fp.split('class')[1].split('.')[0])))
        c += 1
        if c % 100 == 0:
            print(time.time() - time0)
            labels = np.array(img_infos)
            np.save('images.npy', images)
            np.save('labels.npy', labels)
    labels = np.array(img_infos)
    np.save('images.npy', images)
    np.save('labels.npy', labels)

def image_load(filename_image, filename_label):
    images = np.load(filename_image) / 255
    labels = np.load(filename_label)
    return images, labels

def classifier_ResNet50(images):
    model = ResNet50()
    model.compile(
        optimizer="rmsprop"
    )
    model.fit(
        x=images,
        y=labels,
        batch_size=None,
        epochs=1,
        verbose=1,
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    )
    model.evaluate(
    x=images,
    y=labels,
    batch_size=None,
    verbose=1,
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    return_dict=False,
    )
    c = 0
    predictions = dict()
    for img in images:
        image = np.pad(img, ((12,12),(12,12),(0,0)), 'constant',constant_values=((0, 0), (0, 0), (0, 0)) )
        print(image.shape)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = piResNet50(image)
        prediction = model.predict(image)
        prediction = DPResNet50(prediction, top=3)[0]
        print(prediction)
        predictions[c] = prediction
        c += 1
    return predictions
# # pgd_attack = torchattacks.PGD(resNet, eps = 0.3)
# # print(3)
# # adversarial_images = pgd_attack(images, labels)
# # print(4)
# # np.save('adversarial_images.npy', adversarial_images)
# # print(5)
# #

# read_files(directory)
images, labels = image_load(filename_image, filename_label)
predictions = classifier_ResNet50(images)
print('Predictions: %d' % len(predictions))
# save to file
dump(predictions, open('ResNetPredictions.pkl', 'wb'))
