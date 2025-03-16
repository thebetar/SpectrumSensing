# Introduction

This repository is related to my thesis project where I will attempt to predict occupancy of certain frequencies using generative AI.

# How to run

To run this project jupyter notebook is used which runs python in behind a clear graphical user interface. The dependencies needed to run everything are

-   Jupyter notebook (https://jupyter.org/) - follow steps to install or when using VSCode install the extension and you are ready to go
-   Pytorch (https://pytorch.org/get-started/locally/) - follow steps in Pytorch documentation to download the proper version for system specs
-   h5py (https://www.h5py.org/) - run `pip install h5py`
-   torchviz (https://github.com/szagoruyko/pytorchviz) - run `pip install torchviz`
-   matplotlib (https://matplotlib.org/) - run `pip install matplotlib`

After installing this open one of the models in `src/{MODEL}_model.ipynb` and run the file

# Description

The aim of this project will be to take recordings of different frequencies and meassuring the occupancy and comparing this to multiple chosen features. Then an AI model will be trained on this model to make predictions on future occupancy of a certain frequency creating a more efficient way to find frequencies with low occupancy. This will aid in situations where this is high interference from other radio signals, for instance for Wi-Fi in a crowded urban area like an apartment block.

The finished model will take as an input the current occupancy of multiple frequencies and some other features, currently we think this would be time, geolocation and date. The model will then predict the occupancy of each frequency and return this as a result providing an easy way to find the best frequency to form a connection over due to it's low occupancy rate.

## Convolutional layer

A convolutional layer is used to find patterns and relations between data. The data can be changed using a FFT but the neural network should be able to figure out the relations by itself using convolution. Another example of convolution is that the model can tweak the weights of it's output to get an optimal reading.

## LTSM model

The long short term memory model was chosen for it's effectiveness for predicting future observations based on prior knowledge. It takes the last couple of values to predict the next value. For our purpose this should work well.

## SVM model

After reading some papers I found out that SVM models can work very well to classify occupancy channels from a sequence of IQ samples. For this reason I have also included this model. It's simplicity makes it a great pick to play around with. SVM is also combined with simple convolution to increase accuracy.

# Notes

Some additional information for understanding the taken approach and gotten results.

-   Signal strength is meassured in dBM, it is impacted by distance to the source and interference of other signals at the same frequency
-   Signal strengths are meassured starting from 0 which is perfected with zero interference and distance, which is not possible to achieve to a big negative number, overall -50dBM or higher is considered excellent then till -67dBM is considered good, below this and problems start to arise and below -80dBM connection will become unreliable.
-   Signal to noise ratio is described in dB, for instance 10dB means that the signal is 10 times stronger than the noise. So the higher the better the signal is received.

# Resources

## Informational videos

-   https://www.youtube.com/watch?v=AsNTP8Kwu80
-   https://www.youtube.com/watch?v=YCzL96nL7j0
-   https://www.youtube.com/watch?v=HGwBXDKFk9I

## Repositories

-   https://github.com/Darth-Kronos/Spectrum-Sensing
-   https://github.com/vineeths96/Spectrum-Sensing-for-Cognitive-Radio
-   https://github.com/wineslab/deepsense-spectrum-sensing-datasets

## Data sets

-   https://www.kaggle.com/datasets/suraj520/rf-signal-data
-   https://www.kaggle.com/datasets/suraj520/cellular-network-analysis-dataset

## Papers

-   https://www.nvidia.com/en-us/glossary/generative-ai/
-   https://doi.org/10.1186/s13638-020-01870-7
-   https://www.elastic.co/what-is/large-language-models#what-is-the-difference-between-large-language-models-and-generative-ai
-   https://arxiv.org/pdf/1804.00709
-   https://www.mdpi.com/2079-9292/13/14/2705

## Tools

-   https://www.gnuradio.org/
-   https://pytorch.org/

# Todo

-   Add spectrogram images to thesis document to describe better what data went in and what prediction came out
-   Add formulas using latex to document to describe how LSTM, SVM, convolution, etc. works.
