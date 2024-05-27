# Cancer Detection using YOLO

The model is trained on the "Crowds Cure Cancer" dataset
 

### Getting Started
`pip install -r deps.txt`

### Train


#### Downloading and cleaning the data
The data must be downloaded directly from [Kaggle](https://www.kaggle.com/kmader/crowds-cure-cancer-2017), where you need to create a username and password (if you don't already have one) in order to download the dataset. Once you have downloaded and unzipped the dataset, you will have the raw images and CSV data. We clean the CSV data down to only the necessary information using the `clean_data.py` script in the `label_data/` directory, which produces a new, clean CSV file which is used in the training and example usage usage of the model.

- Train:

`$ python model.py`

### Test
`$ python predict.py`


### Directory Structure

```
| $PROJECT_ROOT/
|---| README.md
|---| crowds-cure-cancer-2017/ ## <-- result of Kaggle download
|---|---| annotated_dicoms.zip
|---|---| compressed_stacks.zip
|---|---| CrowdsCureCancer2017Annotations.csv ## <---- @move this file to $PROJECT_ROOT/YOLO/label_data
|---| data/
|---|---| TCGA-09-0364
|---|---| ...
## put the data files into this 'data/' directory
|---|---| ...
|---|---| TCGA-OY-A56Q
|---| YOLO/
|---|---| label_data/
|---|---|---| clean_data.py
|---|---|---| CCC_clean.py
|---|---|---| CrowdsCureCancer2017Annotations.csv ## <---- @Here
|---|---| model.py
|---|---| predict.py
|---|---| deps.txt
|---|---| trained_model/
```

## Aknowledgements
* The "Crowds Cure Cancer" dataset used to train the model in this repo can be found on Kaggle [here](https://www.kaggle.com/kmader/crowds-cure-cancer-2017)
* The YOLO algorithm used in this project was developed by Redmond et. al. is described in paper found [here](https://arxiv.org/pdf/1506.02640.pdf) 
