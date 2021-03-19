# RDGCN replication and application to SRSPRS(DBP-YG)


##0. Setup

###0.1 Installing the requirements:
Please install the packages needed via:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

###0.2 Downloading the data
Download the original embedding files (the .json files) from this Google Drive directory:

``
https://drive.google.com/drive/folders/13u-4r4aJbjhUPRbDXrVFA3QfQS0y_8Ye
``

Kindly place the embedding files in the respective dataset folders.
The SRSPRS dataset does not need to be downloaded, it comes with the git repository.

###0.3 Downloading the embedding model:
Download the glove.840b.300d embedding model from:

``
https://nlp.stanford.edu/projects/glove/
``

Kindly place the embedding .txt file in the top level folder of the project.

###0.4 Creating the embedding file for the SRSPRS dataset:
This step is not required for any of the DBP15K datasets. To create the embedding file for the SRSPRS dataset, run:

``
python include/Vectorization.py
``

##1. Usage

###1.1 Replication
To replicate the results on one of the datasets, run the following command with the dataset specified as language:

``
python main.py --language=fr_en
``

###1.2 Application on the SRPRS dataset:
Application on any dataset works the same. After generating the embedding file for the SRPRS dataset (point 0.4), \
run the following command (with the dataset specified as language):

``
python main.py --language=dbp_yg
``