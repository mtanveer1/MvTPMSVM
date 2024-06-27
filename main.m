clc;
clear

data=load('aus.mat');
XA=data.X1;
XB=data.X2;
y=data.y;
[m1,~]=size(XA);
for i=1:m1
    if y(i)==0
        y(i)=-1;
    end
end

XA=[XA,y];
XB=[XB,y];

% Cross validation (train: 70%, test: 30%)
cv = cvpartition(m1,'HoldOut',0.3);
idx = cv.test;
% Separate to training and test data
XA_train = XA(~idx,:);
XA_test  = XA(idx,:);
XB_train = XB(~idx,:);
XB_test  = XB(idx,:);

sig_best=1;
c1_best=1;


[Test_accuracy, accuracy_viewA, accuracy_viewB,recall,precision,F_1score,G_means, Test_time] = MvTPMSVM(XA_train, XB_train, XA_test, XB_test, c1_best, c1_best, c1_best,c1_best,c1_best,c1_best, sig_best,sig_best,0.1);