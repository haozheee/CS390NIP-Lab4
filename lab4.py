# CS390-NIP GAN lab
# Max Jacobson / Sri Cherukuri / Anthony Niemiec
# FA2020
# uses Fashion MNIST https://www.kaggle.com/zalando-research/fashionmnist
# uses CIFAR-10 https://www.cs.toronto.edu/~kriz/cifar.html

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.optimizers import Adam, SGD
from skimage.transform import rescale, resize
# from scipy.misc import imsave
import imageio
from skimage.io import imread_collection
import random

import matplotlib.pyplot as plt

random.seed(1618)
np.random.seed(1618)
tf.compat.v1.set_random_seed(1618)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Modify this ratio to adjust the traning epochs for the generator and the discriminator, by Haozhe
RATIO = 0.5

# NOTE: mnist_d is no credit
# NOTE: cifar_10 is extra credit
#DATASET = "mnist_d"
DATASET = "mnist_f"
#DATASET = "cifar_10"
#DATASET = "warcraft"

if DATASET == "mnist_d":
    IMAGE_SHAPE = (IH, IW, IZ) = (28, 28, 1)
    LABEL = "numbers"

elif DATASET == "mnist_f":
    IMAGE_SHAPE = (IH, IW, IZ) = (28, 28, 1)
    CLASSLIST = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    # TODO: choose a label to train on from the CLASSLIST above
    LABEL = "dress"

elif DATASET == "cifar_10":
    IMAGE_SHAPE = (IH, IW, IZ) = (32, 32, 3)
    CLASSLIST = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    LABEL = "airplane"

if DATASET == "warcraft":
    IMAGE_SHAPE = (IH, IW, IZ) = (32, 32, 3)
    LABEL = "none"

IMAGE_SIZE = IH*IW*IZ

NOISE_SIZE = 100    # length of noise array

# file prefixes and directory
OUTPUT_NAME = DATASET + "_" + LABEL
OUTPUT_DIR = "./outputs/" + OUTPUT_NAME

# NOTE: switch to True in order to receive debug information
VERBOSE_OUTPUT = False

################################### DATA FUNCTIONS ###################################

# Load in and report the shape of dataset
def getRawData():
    if DATASET == "mnist_f":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.fashion_mnist.load_data()
    elif DATASET == "cifar_10":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
    elif DATASET == "mnist_d":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
    elif DATASET == 'warcraft':
        wc3images = imread_collection("./wc3avatars/*.png", conserve_memory=True)
        wc3images = [resize(img, (IH, IW, IZ), anti_aliasing=True) for img in wc3images]
        sp = int(0.8*len(wc3images))
        xTrain = np.array(wc3images[:sp], dtype=float)
        yTrain = np.ones_like(xTrain)
        xTest = np.array(wc3images[sp:], dtype=float)
        yTest = np.ones_like(xTest)

    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))

# Filter out the dataset to only include images with our LABEL, meaning we may also discard
# class labels for the images because we know exactly what to expect
def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if DATASET == "mnist_d":
        xP = np.r_[xTrain, xTest]
    elif DATASET == "warcraft":
        xP = np.r_[xTrain, xTest]
    else:
        c = CLASSLIST.index(LABEL)
        x = np.r_[xTrain, xTest]
        y = np.r_[yTrain, yTest].flatten()
        ilist = [i for i in range(y.shape[0]) if y[i] == c]
        xP = x[ilist]
    # NOTE: Normalize from 0 to 1 or -1 to 1
    #xP = xP/255.0
    xP = xP/127.5 - 1
    print("Shape of Preprocessed dataset: %s." % str(xP.shape))
    return xP


################################### CREATING A GAN ###################################

# Model that discriminates between fake and real dataset images
def buildDiscriminator():
    model = Sequential()

    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=IMAGE_SHAPE))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # Creating a Keras Model out of the network

    inputTensor = Input(shape = IMAGE_SHAPE)
    return Model(inputTensor, model(inputTensor))

# Model that generates a fake image from random noise
def buildGenerator():
    model = Sequential()
    model.add(Dense(IMAGE_SIZE, input_dim=NOISE_SIZE))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Reshape((IW, IH, IZ)))

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(IZ, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    model.add(Reshape(IMAGE_SHAPE))

    # Creating a Keras Model out of the network
    inputTensor = Input(shape = (NOISE_SIZE,))
    return Model(inputTensor, model(inputTensor))


# Clarification on the definition of arguments:
# ratio = generator_epochs / discriminator_epochs
# discriminator_epochs = epochs of the discriminator
# by Haozhe
def buildGAN(images, discriminator_epochs=40000, ratio=1, batchSize = 32, loggingInterval = 0):
    # initialize loss records used for plotting.
    discriminator_loss_values = []
    generator_loss_values = []

    # Setup
    opt = Adam(lr = 0.001)
    loss = "binary_crossentropy"

    # Setup adversary
    adversary = buildDiscriminator()
    adversary.compile(loss = loss, optimizer = opt, metrics = ["accuracy"])

    # Setup generator and GAN
    adversary.trainable = False                     # freeze adversary's weights when training GAN
    generator = buildGenerator()                    # generator is trained within GAN in relation to adversary performance
    noise = Input(shape = (NOISE_SIZE,))
    gan = Model(noise, adversary(generator(noise))) # GAN feeds generator into adversary
    gan.compile(loss = loss, optimizer = opt)

    # Training
    trueCol = np.ones((batchSize, 1))
    falseCol = np.zeros((batchSize, 1))
    for epoch in range(discriminator_epochs):
            # A. Train Discriminator

            # Train discriminator with a true and false batch
            batch = images[np.random.randint(0, images.shape[0], batchSize)]
            noise = np.random.normal(0, 1, (batchSize, NOISE_SIZE))
            genImages = generator.predict(noise)
            advTrueLoss = adversary.train_on_batch(batch, trueCol)
            advFalseLoss = adversary.train_on_batch(genImages, falseCol)
            advLoss = np.add(advTrueLoss, advFalseLoss) * 0.5

            # B. Train Generator
            ig = 1
            genLoss = 0
            while ig <= ratio:
                ig = ig + 1
                # Train generator by training GAN while keeping adversary component constant
                noise = np.random.normal(0, 1, (batchSize, NOISE_SIZE))
                genLoss_i = gan.train_on_batch(noise, trueCol)
                genLoss = genLoss + genLoss_i
            genLoss = genLoss/ratio

            discriminator_loss_values.append(advLoss[0])
            generator_loss_values.append(genLoss)

            # Logging
            if loggingInterval > 0 and epoch % loggingInterval == 0:
                print("\tEpoch %d:" % epoch)
                print("\t\tDiscriminator loss: %f." % advLoss[0])
                print("\t\tDiscriminator accuracy: %.2f%%." % (100 * advLoss[1]))
                print("\t\tGenerator loss: %f." % genLoss)
                runGAN(generator, OUTPUT_DIR + "/" + OUTPUT_NAME + "_test_%d.png" % (epoch / loggingInterval))
                # save plot of loss values over training steps
                line_dis, = plt.plot(range(len(discriminator_loss_values)), discriminator_loss_values, label='discriminator loss')
                line_gen, = plt.plot(range(len(generator_loss_values)), generator_loss_values, label='generator loss')
                plt.legend(handles=[line_dis, line_gen])
                plt.savefig(OUTPUT_DIR + "/" + OUTPUT_NAME + "_training_plot.pdf")
                with open("./discriminator_loss_values.txt", "w") as wf:
                    wf.writelines([str(x) for x in discriminator_loss_values])
                with open("./generator_loss_values.txt", "w") as wf:
                    wf.writelines([str(x) for x in generator_loss_values])

    return (generator, adversary, gan), (discriminator_loss_values, generator_loss_values)

# Generates an image using given generator
def runGAN(generator, outfile):
    noise = np.random.normal(0, 1, (1, NOISE_SIZE)) # generate a random noise array
    img = generator.predict(noise)[0]               # run generator on noise
    img = np.squeeze(img)                           # readjust image shape if needed
    img = (0.5*img + 0.5)*255                       # adjust values to range from 0 to 255 as needed
    # imsave(outfile, img)                            # store resulting image
    imageio.imwrite(outfile, img)


################################### RUNNING THE PIPELINE #############################

def main():
    print("Starting %s image generator program." % LABEL)
    # Make output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Receive all of mnist_f
    raw = getRawData()
    # Filter for just the class we are trying to generate
    data = preprocessData(raw)
    # Create and train all facets of the GAN
    (generator, adv, gan), (disLoss, genLoss) = buildGAN(data, discriminator_epochs=50000, ratio=2, loggingInterval=5000)

    # save plot of loss values over training steps
    line_dis, = plt.plot(range(len(disLoss)), disLoss, label='discriminator loss')
    line_gen, = plt.plot(range(len(genLoss)), genLoss, label='generator loss')
    plt.legend(handles=[line_dis, line_gen])
    plt.savefig(OUTPUT_DIR + "/" + OUTPUT_NAME + "_training_plot.pdf")
    with open("./discriminator_loss_values.txt", "w") as wf:
        wf.writelines([str(x) for x in disLoss])
    with open("./generator_loss_values.txt", "w") as wf:
        wf.writelines([str(x) for x in genLoss])


    # Utilize our spooky neural net gimmicks to create realistic counterfeit images
    for i in range(10):
        runGAN(generator, OUTPUT_DIR + "/" + OUTPUT_NAME + "_final_%d.png" % i)
    print("Images saved in %s directory." % OUTPUT_DIR)

if __name__ == '__main__':
    main()