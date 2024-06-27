
 # MvTPMSVM: Multi-view learning with twin parametric margin SVM

Please cite the following paper if you are using this code.

A. Quadir, M. Tanveer (2024). “Multi-view learning with twin parametric margin SVM, Neural Networks, Elsevier” 

The experiments are executed on a computing system possessing Matlab2023a software, an Intel(R) Xeon(R) CPU E5-2697 v4 processor operating at 2.30 GHz with 128-GB Random Access Memory (RAM), and a Windows-11 operating platform.

We have deployed a demo of the 'MvTPMSVM' model using the 'aus' dataset.

The following are the best hyperparameters set with respect to the “aus” dataset

Regularization Parameter c1_best=2,  sig_best=1

Description of files:

main.mat: This is the main file to run selected algorithms on datasets. In the path variable specificy the path to the folder containing the codes and datasets on which you wish to run the algorithm.

MvTPMSVM.mat: This file contains the function to solve the optimization problem

judgement.mat: Function to evaluate the accuracy

kernelfunction: This function corresponds to the kernel function mapping of data points.

For a comprehensive understanding of the experimental setup, please refer to the paper. Should you encounter any bugs or issues, feel free to contact A. Quadir at mscphd2207141002@iiti.ac.in.



           

