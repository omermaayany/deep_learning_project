# Deep Learning Project – RL + LLM 

This repository contains the main code used to run the experiments in two **Google Colab** notebooks:
1. One notebook located in the **root directory** (for the LLM models)
2. One notebook inside the **RL** folder (for the reinforcement learning module)

---

Setup Instructions

1. **Connect to Google Drive** using the following link:  
   ➤ https://drive.google.com/drive/folders/1OBRXBehcxhPZ2TvTcFxc7FoSOYpZM0Ph?usp=sharing

   The Drive folder contains:
   - Pretrained models required for running the experiments  
   - Relevant runtime environments and dependencies  

   Make sure all environments are properly installed and mounted before running the notebooks.

---

Execution Order

1. **Run the RL notebook first.**  
   This notebook prepares the reinforcement learning environment and generates the predicted human actions.

2. **Run the LLM notebook second.**  
   This notebook loads the selected language model and produces responses based on the RL predictions.

3. In the LLM notebook, **run each model separately.**  
   Update the model name in the marked code cells according to the model you want to test.

---

Documentation

A detailed explanation of the project objectives, implementation, model architecture, and experimental results  
can be found in the attached **PDF report**.

