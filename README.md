# Disaster_Response_Pipeline
This project is to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a multilabel classification model and a Flask backend and webpage that classifies disaster messages.
### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


# Installation<a name="installation"></a>
The jupyter notebook file should be run with python 3.X . Need to install nltk and download associate files.

# Project Motivation<a name="motivation"></a>
Built a multilabel classifier to do disaster messages classification by a subset of labels, so the messages can be delivered to the departments associated to response the disaster labels.


# File Descriptions<a name="files"></a>
There are 3 python file implement the product.
process_data.py--preprocess the original message and category file and store the data in sqlite file database.
train_classifier.py--process text and built the multilabel classification model and store the model in a pkl file.
run.py--flask backend file
templates--frontend go.html and master.html for web access.

# Results<a name="results"></a>
The disaster message multilabel classification results can be accessed by http service with Flask backend.

# Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to [Figure Eight](https://www.figure-eight.com/) for the data.And [Udacity](http://www.udacity.com) for project design and instructions.
