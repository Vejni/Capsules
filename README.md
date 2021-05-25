# Capsules Thesis

## Overview

This repository contains the files for my L4 project on Capsule Networks and their use in breast cancer classification. The project has been written and test in Python 3.8.8, and all other packages can be found in requirements.txt, which can simply be installed via pip or conda. 

## Project Structure

- Root: The root folder contains project speficic files, but also the main script which can be run locally. 
- src: Contains the source code, such as definitions for the patch- and image-wise networks, training loops and methods for dataset manipulation. Additionally, it contains truncated code for the 3 types of capsules used in the project: DynamicCaps, VarCaps and SRCaps, which can be found in the appropriate folders.
- notebooks: Contains Jupyter Notebook files from the early days of the project. They are there for legacy and should not be marked or run. 
- models: This folder contains saved models, as well as saved checkpoints. Additionally the outs folder contains outputs from experiments, which were then used to create plots. These files are NOT unaltered, as some have been trimmed or put together for ease of plotting.
- docs: Contains the presentation, report and poster files.
- data: Folder for datasets. It is recommended to create the datasets here using the appropriate script.

## How to use

To run this project it is recommended to do so on Google Colab or perhaphs Kaggle. The steps in the user manual describe how to set up the system on Google Colab, and import the datasets. If one would like to set this up locally, they need to obtain the datasets and then recreate the splits and patches using the datasets.py script in the src folder.