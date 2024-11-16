# Introduction

This repository is related to my thesis project where I will attempt to predict occupancy of certain frequencies using generative AI.

# Description

The aim of this project will be to take recordings of different frequencies and meassuring the occupancy and comparing this to multiple chosen features. Then an AI model will be trained on this model to make predictions on future occupancy of a certain frequency creating a more efficient way to find frequencies with low occupancy. This will aid in situations where this is high interference from other radio signals, for instance for Wi-Fi in a crowded urban area like an apartment block.

The finished model will take as an input the current occupancy of multiple frequencies and some other features, currently we think this would be time, geolocation and date. The model will then predict the occupancy of each frequency and return this as a result providing an easy way to find the best frequency to form a connection over due to it's low occupancy rate.

# Notes

Some additional information for understanding the taken approach and gotten results.

-   Signal strength is meassured in dBM, it is impacted by distance to the source and interference of other signals at the same frequency
-   Signal strengths are meassured starting from 0 which is perfected with zero interference and distance, which is not possible to achieve to a big negative number, overall -50dBM or higher is considered excellent then till -67dBM is considered good, below this and problems start to arise and below -80dBM connection will become unreliable.
-   Signal to noise ratio is described in dB, for instance 10dB means that the signal is 10 times stronger than the noise. So the higher the better the signal is received.

# Resources

## Informational videos

-   https://www.youtube.com/watch?v=AsNTP8Kwu80
-   https://www.youtube.com/watch?v=YCzL96nL7j0

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

-   Display the occupancy by frequency and time in a heatmap
