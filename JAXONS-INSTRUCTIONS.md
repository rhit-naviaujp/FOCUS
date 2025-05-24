# The FOCUS Framework - Jaxon's README
I did not add any new code to this project, however I will explain how I evaluated the model using the demo script provided. If any problems arise setting up the environment or installing packages, consult the FOCUS README. 

## Step 1
First, you should ssh into the gebru server. Create a conda environment by issuing the command `conda create --name focus python=3.8`.
 
NOTE: If Python version 3.8 is not installed, install it first using this link https://www.python.org/downloads/release/python-380/

## Step 2
Now that your environment has been created, activate the environment with the command `conda activate focus`. You should then clone a forked version of the FOCUS repository onto the server.

## Step 3
You will need to install PyTorch before installing the requirements.txt file. Do so by running this command `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`. This command installs PyTorch with a necessary build for CUDA 12.1.

## Step 4
You should now be ready to install the remaining requirements. Do so by writing the command inside your FOCUS repository `pip install -r requirements.txt`

## Step 5
Now you should install detectron2 and other dependencies. Type the following commands in order:
`git clone https://github.com:facebookresearch/detectron2.git`
`cd detectron2 && pip install -e . && cd ..`
`pip install git+https://github.com/cocodataset/panopticapi.git`
`cd third_party/CLIP`
`python -m pip install -Ue .`
`cd ../../`
`cd focus/modeling/pixel_decoder/ops && sh make.sh && cd ../../../../`

## Step 6
Now that all dependencies have been installed, create a folder called `model_weights` inside your FOCUS repository. If you do not have gdown installed already, install it using the command `pip install gdown`. This a link to a shared drive with a folder containing all of the model weights: https://drive.google.com/drive/folders/1IcyZnqc4vcsvSUcKb2llYGPt3ClFGjPl?usp=drive_link
Navigate to the `model_weights` directory and run the following command `gdown --folder 'https://drive.google.com/drive/folders/1IcyZnqc4vcsvSUcKb2llYGPt3ClFGjPl?usp=drive_link'. You may get the following error:
![image](https://github.com/user-attachments/assets/2c73b5eb-a4e7-4bc1-a820-919417d78893)
If this happens, try waiting until after midnight to run the gdown command. This command retrieves the FOCUS model weights to run inference. 

## Step 7
We are almost ready to test the model on images. Use the links provided in the FOCUS `README.md` under the `Prepare Datasets` section to download required datasets. I found the best way to download these datasets is to copy the URL link to the file, and run the command `gdown --fuzzy <URL_LINK_TO_DATASET>` inside of the `datasets` directory. Then you can unzip these folders by running the command `unzip filename.zip`. 

## Step 7.5
You do not need to download all datasets to run the demo script and test individual images. Some of the dataset links lead to pages that are not in English, so it was difficult for me to determine which buttons were links to datasets. Additionally, I could not find a way to copy the link to the dataset zip folder under these links. Two datasets that worked well for me are `CAMO` and `ECSSD`. 

## Step 8
Navigate back to your FOCUS directory. Run the following commands to prepare the `CAMO` and `ECSSD` datasets
`python utils/prepare/prepare_camo.py`
`python utils/prepare/prepare_ecssd.py`

## Step 9
It's time to run our demo! Do this by using the following command:
```
python demo/demo.py --config-file path/to/your/config \
  --input path/to/your/image \
  --output path/to/your/output_file \
  --opts MODEL.WEIGHTS path/to/your/weights
```

## Disclaimer
I believe the reason I could not run the `Evaluation` step you see in the FOCUS `README.md` is because I could not download all of the datasets they ask for. I tried running evaluation with only a couple of the datasets and got the following error:
`FileNotFoundError: [Errno 2] No such file or directory: 'datasets/NC4K/NC4K_SEMANTIC.json'`
