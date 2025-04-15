# CPEN455 2024 W2 Course Project: Conditional PixelCNN++ for Image Classification


This repo is for CPEN 455 course project 2024 Winter Term 2 at UBC. **The goal of this project is to implement the Conditional PixelCNN++ model and train it on the given dataset.** After that, the model can both generate new images and classify the given images. **For grading, we evaluate the model based on both the generation performance and classification performance.**

## Overview

- [CPEN455 2024 W2 Course Project: Conditional PixelCNN++ for Image Classification](#cpen455-2024-w2-course-project-conditional-pixelcnn-for-image-classification)
  - [Overview](#overview)
  - [Project Introduction](#project-introduction)
  - [Basic tools](#basic-tools)
  - [Running original PixelCNN++ code](#running-original-pixelcnn-code)
  - [Detailed Guidance](#detailed-guidance)
  - [Dataset](#dataset)
  - [Submission Requirements:](#submission-requirements)
  - [Model Evaluation](#model-evaluation)
  - [Grading Rubric](#grading-rubric)
    - [Bonus Points:](#bonus-points)
  - [Milestones](#milestones)
  - [Final project report guidelines](#final-project-report-guidelines)
    - [Report Length and Structure:](#report-length-and-structure)
    - [Model Presentation Tips:](#model-presentation-tips)
    - [Experiments Section:](#experiments-section)
    - [Conclusion Section:](#conclusion-section)
  - [Academic Integrity Guidelines for the Course Project](#academic-integrity-guidelines-for-the-course-project)

## Project Introduction

PixelCNN++ is a powerful generative model with tractable likelihood. It models the joint distribution of pixels over an image $x$ as the following product of conditional distributions.

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/-jZg8HEMyFnpduNsi-Alt.png" width = "500" align="center"/>

where $x_i$ is a single pixel.

Given a class embedding $c$, PixelCNN++ can be extended to conditional generative tasks following:

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/_jv7O2Z_1s1oYLXjIqS1V.png" width = "260" align="center"/>

In this case, with a trained conditional PixelCNN++, we could directly apply it to the zero-shot image classification task by:

<img src="https://cdn-uploads.huggingface.co/production/uploads/65000e786a230e55a65c45ad/P4co1MxbW8tmhgYwBNOxk.png" width = "350" align="center"/>

**Task:** For our final project, you are required to achieve the following tasks
* We will provide you with codes for an unconditional PixelCNN++. You need to adapt it to conditional image generation task and train it on our provided database.

## Basic tools
We recommend the following tools for debugging, monitoring, and training process:
<details>
  <summary>
    Wandb
  </summary>
Wandb is a tool that helps you monitor the training process. You can see the loss, accuracy, and other metrics in real time. You can also see the generated images and the model structure. You can find how to use wandb in the following link: https://docs.wandb.ai/quickstart
</details>

<details>
  <summary>
    Tensorboard
  </summary>
Tensorboard is another tool that helps you monitor the training process. You can find how to use tensorboard in the following link: https://www.tensorflow.org/tensorboard/get_started
</details>

<details>
  <summary>
    PDB
  </summary>
PDB is an interactive Python debugger. You can use it to debug your code. You can find how to use pdb in the following link: https://docs.python.org/3/library/pdb.html
</details>

<details>
  <summary>
    Conda
  </summary>
Conda is a package manager. You can use it to create a virtual environment and install the required packages. You can find how to use conda in the following link: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
</details>


## Running original PixelCNN++ code
<details>
  <summary>
    Running on Google Colab
  </summary>
  <p>Please refer to <a href="https://colab.research.google.com/drive/190mJ4_iPozG8W__bjjQ7iJ8fNxcO48w5?usp=sharing" target="_blank">this Colab notebook</a> to run the project using Google Colab. You need to follow the following steps:</p>
  <ol>
    <li>Download the project, and place it in your Google Drive in a folder <code>cpen455-project</code>. Make sure this folder is in the root of your Google Drive.</li>
    <li>Run the notebook, which will mount your Google Drive folder into the notebook. This will run the original PixelCNN++ code.</li>
  </ol>

  <p><strong>Note 1:</strong> Make sure you are using a GPU-enabled Colab, otherwise, the code will run very slowly!</p>

  <p><strong>Note 2:</strong> You can access the project files using the sidebar that Colab provides, and click on the desired file you would like to modify. Since the files are in your Google Drive, the changes will be persistent.</p>

  <p><strong>Note 3:</strong> If you are using this method, we recommend taking frequent snapshots of your code as you progress (e.g., by downloading the code), since this method does not support Git or any version control.</p>
</details>

<details>
  <summary>
    Running Locally on Linux/Mac/Windows
  </summary>
we provided the code for the PixelCNN++ model. Before you run the code, you need to install the required packages by running the following command:
  
```
conda create -n cpen455 python=3.10.13
conda activate cpen455
conda install pip3
```

when you type the command `which pip3`, you should see the path of the pip3 in the virtual environment but not in the system pip3 path.

if you make sure the pip3 is in the virtual environment, you can install pytorch via this touorial: https://pytorch.org/get-started/locally/, you should choose the right version of command according to your system. For example, if you use linux with cuda support, you should use the following command:
```
pip3 install torch torchvision torchaudio
```
After you install the pytorch, you can install the other required packages by running the following command:
```
pip install -r requirements.txt
```

Please note that we guarantee that the requirements.txt file includes all the Python packages necessary to complete the final project. Therefore, **please DO NOT install any third-party packages.** If this results in the inability to run the submitted code later on, you may need to take responsibility. If you have any questions regarding Python packages, please contact the teaching assistant.

And then, you can run the code via the following command:
```
python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 16 \
--sampling_interval 25 \
--save_interval 25 \
--dataset cpen455 \
--nr_resnet 1 \
--lr_decay 0.999995 \
--max_epochs 100 \
--en_wandb True \
```

If you don't want to enable wandb, run the code via the following command:
```
python pcnn_train.py \
--batch_size 16 \
--sample_batch_size 16 \
--sampling_interval 25 \
--save_interval 25 \
--dataset cpen455 \
--nr_resnet 1 \
--lr_decay 0.999995 \
--max_epochs 100
```

For Windows users, you can run the code via the following command, please note that you should remove the `\` at the end of each line:
```
python pcnn_train.py --batch_size 16 --sample_batch_size 16 --sampling_interval 25 --save_interval 25 --dataset cpen455 --nr_resnet 1 --lr_decay 0.999995 --max_epochs 100 --en_wandb True
```
</details>


If you want to go into more detail about Pixelcnn++, you can find the original paper at: https://arxiv.org/abs/1701.05517

You can also refer to the following implementations of PixelCNN++:

1. Original PixelCNN++ repository implemented by OpenAI: https://github.com/openai/pixel-cnn

2. Pytorch implementation of PixelCNN++: https://github.com/pclucas14/pixel-cnn-pp

## Detailed Guidance

Please refer to the [guidance.md](guidance/guidance.md) file for detailed guidance on the project.

## Dataset

In the code base we provided, we have included the data required to train conditional PixelCNN++. The directory structure is as follows:
```
data
├── test
├── train
└── validation
```

The `train` directory contains 4160 labeled training images, divided into four categories. The `validation` directory contains 520 labeled validation images. The `test` directory contains 520 unlabeled testing images used for evaluating model performance.


Ground truth labels for the training set and validation set are stored in the `data/train.csv` and `data/validation.csv` respectively. These two `.csv` files contain two columns: `id` and `label`, as shown below:

```
id, label
0000000.jpg,1
0000001.jpg,0
0000002.jpg,3
0000003.jpg,1
0000005.jpg,3
0000006.jpg,3
0000007.jpg,2
0000008.jpg,0
```

## Submission Requirements:
You must compress the following materials into a zip file and submit it on Canvas:

- [ ] **Complete project code**
  - [ ] **Implement two evaluation scripts**:
    - [ ] `generation_evaluation.py`: 
      - Save generated images to `./samples` directory
      - Include 100 total images (25 per class across 4 classes)
      - Detailed instructions are introduced in [generation_evaluation.py](generation_evaluation.py)
    - [ ] `classification_evaluation.py`:
      - Evaluate model accuracy on validation set
      - Maintain original code interfaces (we will test using the test set)
      - Detailed instructions are introduced in [classification_evaluation.py](classification_evaluation.py)
    - **Important**: Avoid attempts to circumvent proper evaluation. Submissions that generate invalid samples or fail classification tasks while containing deceptive code **will receive penalties**. If struggling with implementation, refer to [Milestone and Grading](#milestone-and-grading) for partial credit options rather than non-functional workarounds.
    - **Important**: Please DO NOT change any other definitions in the two interfaces `generation_evaluation.py` and `classification_evaluation.py`. All you can modify is within "begin of your code" and "end of your code". As for other parts of this repo, you can modify them arbitrarily.
  - [ ] **Model checkpoint**:
    - Save to `models/conditional_pixelcnn.pth`

- [ ] **Project report**
  - The requirement is introduced in [Final project report guidelines](#final-project-report-guidelines)

## Model Evaluation

For the evaluation of model performance, we assess the quality of images generated by conditional PixelCNN++ and the accuracy of classification separately. 

For classification accuracy, we evaluate using **accuracy**. We have provided an evaluation interface function in `classification_evaluation.py`. While you can only validate on the validation set, we will use the test set for final grading. A Hugging Face leaderboard will be provided before the final project deadline so you can submit your classification results and test their performance on the test set.

For assessing the quality of generated images, we have provided an evaluation interface function in `generation_evaluation.py` using the **FID score** to gauge the quality. After the final project deadline, we will run all submitted code on our system and execute the FID evaluation function. It is essential to ensure that your code runs correctly and can reproduce the evaluation results reported in the project. Failure to do so may result in corresponding deductions.

Evaluation of model performance will affect a portion of the final score, but not all of it. After deadlines, we will attempt to reproduce all submitted code, and any cheating discovered will result in deductions and appropriate actions taken. The quality of the code, the completeness of the project, and the ability to reproduce results will all be considered in determining the final score.

## Grading Rubric
The final score is calculated based on Generation Performance, Classification Performance, Report Quality, and Bonus Points. Worth mentioning is that the performance refers to the performance on the test dataset.

+ **Generation Performance**  
  Evaluated using Fréchet Inception Distance (FID) score of generated images  
  - Lower FID scores correspond to better performance
  - Linear interpolate the grades from 0 to full when the FID goes from 60 to 30: 
    - Score = (60 - FID score) / 30 * {full score of generation performance}, when FID score is between 60 and 30
  - < 30 FID score = full score of generation performance
  - \> 60 FID score = 0 points

+ **Classification Performance**  
  - Linear interpolate the grades from 0 to full when the acc goes from 25% to 75%:  
    - Score = (acc - 0.25) / 0.5 * {full score of classification performance}, when acc is between 25% and 75%
  - Accuracy exceeding 75% qualifies for bonus points (see [Bonus Points section](#bonus-points))

+ **Report Quality**  
  - Well-written, high-quality reports demonstrating deep understanding are preferred  

+ **Bonus Points**  
  - See in [Bonus Points section](#bonus-points)

### Bonus Points:
+ Earn **10% more marks of the full score of project(3% in whole course)** if your model outperforms 75% accuracy on the test set.
+ You may gain **10% more marks of the full score of project(3% in whole course)** for conducting a detailed and interesting analysis of your model or generated results.
+ Other Bonus Points/Questions:
  + Why do masked convolution layers ensure the autoregressive property of PixelCNN++?
  + Implement different fusion strategies in the architecture and compare their performance.
  + Why are the advantages of using a mixture of logistics used in PixelCNN++? (hint: You will get the answer if you go through the sampling function, also the similar philosophy shared in deepseek-v2/v3)
  + Read papers and reproduce their methods for inserting conditions, comparing them with common fusion implementations.
  + Compare the performance of your model with dedicated classifiers (e.g., CNN-based classifiers) trained on the same dataset. Think about the advantages and disadvantages of your model compared with dedicated classifiers.

## Milestones
It's hard to set some specific milestones for the project, because each part of the entire project is closely related to each other. Once you implement the conditional insertion into the model, the remaining parts are more or less straightforward.

However, **if you are too busy to implement the entire project or find this project too challenging**, you can still get partial marks by completing some basic tasks, for example:

+ Properly install the dependencies, run the original PixelCNN++ code, and generate some images. Document your findings in the report.(10% of the full score of project)
+ Introduce the PixelCNN++ model in detail within the report and describe your attempts to implement the conditional PixelCNN++ model. (Graded based on how detailed your description is of the concepts you've learned.)
+ Try to answer the [bonus questions](#bonus-points) mentioned previously. 

**For those who are able to complete the entire project, please ignore this part**.

## Final project report guidelines
Students are required to work on projects individually. All reports must be formatted according to the NeurIPS conference style and submitted in PDF format. When combining the report with source code and additional results, the submission on Canvas portal should be in a zip file format.


### Report Length and Structure:

+ The report should not exceed 4 pages, excluding references or appendices.
+ A lengthy report does not equate to a good report. Don't concern yourself with the length; instead, focus on the coding and consider the report as a technical companion to your code and ideas. We do not award extra points for, nor penalize, the length of the report, whether it be long or short.
+ We recommend organizing the report in a three-section format: Model, Experiments, and Conclusion.

### Model Presentation Tips:

+ Include a figure (created by yourself!) illustrating the main computation graph of the model for better clarity and engagement.
+ Use equations rigorously and concisely to aid understanding.
+ An algorithm box is recommended for clarity if the method is complex and difficult to understand from text alone.
+ Provide a formal description of the models, loss functions etc.
+ Distinguish your model from others using comparative figures or tables if possible.
### Experiments Section:
Including at least one of the following is recommended:
+ Ablation study focusing on specific design choices.
+ Information on training methods, and any special techniques used.
+ Both quantitative and qualitative analysis of experimental results.
### Conclusion Section:
+ Summarize key findings and contributions of your project.
+ Discuss limitations and potential avenues for future improvements and research.

## Academic Integrity Guidelines for the Course Project

In developing your model, you are permitted to utilize any functions available in PyTorch and to consult external resources. However, it is imperative to properly acknowledge all sources and prior work utilized.

Violations of academic integrity will result in a grade of ZERO. These violations include:

1. Extensive reuse or adaptation of existing methods or code bases without proper citation in both your report and code.
2. Use of tools like ChatGPT or Copilot for code generation without proper acknowledgment, including details of prompt-response interactions.
3. Deliberate attempts to manipulate the testing dataset in order to extract ground-truth labels or employ discriminative classifiers.
4. Intentional submission of fabricated results to the competition leaderboard.
5. Any form of academic dishonesty, such as sharing code, model checkpoints, or inference results with other students.

Adhering to these guidelines is crucial for maintaining a fair and honest academic environment. Thank you for your cooperation.


# CPEN455-project
