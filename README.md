# FED-SQUARES
Fed-Squares: Use of Squares-Based Tool for analysis of Federated Learning Simulations

Federated Learning (FL) is a new paradigm on Machine Learning(ML) that permits train an ML model in a distributed way.  Thetraining process is executed at the edge-preserving of the user’sprivacy since the data never leaves the local device that differs fromthe standard paradigm centralized, where the users sent data to acentral server.  The FL scenario deals with a more heterogeneous scenario related to data distribution between clients that are notanalyzed by an engineer as in a centralized paradigm.  In general,statistical techniques such as accuracy, recall, precision, log loss,and the confusion matrix, in visualization, are used to compare themodels and interpret if the training process occurs well, and themodel converged to a defined scenario. However, these techniquesonly give an overall idea about how the data is used and threadedby the model, given similar comparison values between two modelswith very different characteristics.  The SQUARES [1] technique permits a more precise evaluation of the model at the instance level,which allows the analysis, data biases, inspection, outliers, and how the model responds while training to the tested samples. This papershows a SQUARES prototype’s development and evaluation of an FL image classification model’s different scenarios. In this more challenging scenario, we expect to help through the visualization finds some insights and exciting inferences and cases that are impossible only with the standards visualizations and metrics before mentionedand most found on ML benchmarks.

[1] D.  Ren,  S.  Amershi,  B.  Lee,  J.  Suh,  and  J.  D.  Williams.   Squares:Supporting interactive performance analysis for multiclass classifiers.vol. 23, pp. 61–70, 2017.

![Fed-Squares Visualization](https://github.com/tvmsouza/FED-SQUARES/blob/master/images/correct_sample.png?raw=true)

# Testing the solution

- Clone the repo to a workspace
- Open the terminal and go to the folder fed-squares
- run the command: python -m http.server
- Access in a browser: http://localhost:8000/
- Select the 'fed_squares.html' file
- On the screen select one of the tests showed at the paper typing one of the names:
  - metrics_stat500
  - metrics_stat1000_1
  - metrics_stat1000_2
  - metrics_stat3000
 -Press the button analyse
 
 # Generating new cases of test
 
- The Federated simulations are made with the LEAF benchmark so you have to install all the requirements and the LEAF benchmark at the link:
  https://github.com/TalwalkarLab/leaf
- After this you have to preprocess the FEMNIST dataset with the filters mentioned in the paper and save the csv/json files, for this substitute all files on the    original leaf folder by the files present in the leaf folder of this repo.
- After replace the files go to the folder on the original femnist leaf/data/femnist through the terminal and execute:
./preprocess.sh -s niid --sf 0.1 -k 0 -t sample --smplseed 123 --spltseed 123
- After the preprocessing the dataset foto the folder leaf/model and execute:
python main.py -dataset femnist -model cnn
- Concluding the training two files will be generated at the folder leaf/model/metrics:
 - metrics_stat.csv and metrics_stat.json
- Put the files on the folder fed_squares of this project.
- Initialize the server and the fed_sqaures visualization acording to the last session.
- Type the name of files to be analised (example: metrics_stat)

