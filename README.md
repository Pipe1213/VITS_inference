# VITS_inference
This repository indicates the steps to generate audios using pretrained models for tts using the VITS architecture proposed on this repo https://github.com/jaywalnut310/vits by Jaehyeon Kim, Jungil Kong, and Juhee Son.

## 1. Clone repository and dependencies
Clone the original repository: https://github.com/jaywalnut310/vits

``git clone https://github.com/jaywalnut310/vits``

python version = 3.6
Then install all the dependencies from the requirements.txt file (some of then are available using conda install but other only using pip install so i recommend to install them one by one)

Additionally, it is recommended to install the appropriate driver for the cuda version of the system, which can be found at the following link: https://pytorch.org/get-started/locally/

## 2. Download pretrained models

Some pre-trained models are available at the following link:

https://drive.google.com/drive/folders/1Vig65ItCC3nmIDJ0FCKdk6Tof7DI44LN?usp=sharing

Each model has its own file containing the symbols used (symbols.py), this file must be located in the 'text' folder and must be adapted depending on each model.

## 3. Inference

Use the inference scripts provided on this repo to generated new audios, only replace the folder path with yours and the desire text to generated and speaker id.
For the pretrained models provided in this repo '0' correspond to the male voice and '1' to the female.

Inference based on phonemes: https://github.com/Pipe1213/VITS_inference/blob/main/generator_mspho_fabs.py 

Inference based on graphemes: https://github.com/Pipe1213/VITS_inference/blob/main/generator_ms_fabs.py

