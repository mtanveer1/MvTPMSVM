function [accuracy, accuracy_viewA, accuracy_viewB,recall,precision,F_1score,G_means,time]= MvTPMSVM(data1, data2, test_set1, test_set2, CA1, CB1,CA2, CB2,D1 ,D2 , kernelparamA, kernelparamB,epsilon)
kerneltype='rbf';

num_truncation = 6;
global global_options
% obtain the +1 pattern and -1 pattern for each view
XA=data1(:,1:end-1);
XB=data2(:,1:end-1);
y=data2(:,end);
A1 = XA(y == 1, :);
B1 = XA(y == -1, :);
A2 = XB(y == 1, :);
B2 = XB(y == -1, :);

tic
% obtain the number of +1 patterns and -1 patterns
l_p = size(A1, 1);
l_n = size(B1, 1);
E_n = eye(l_n);
E_p = eye(l_p);
O_p = zeros(l_p);
O_n=zeros(l_n);

% create the quadratic matrix
P1=kernelfunction(kerneltype, A1, A1, kernelparamA);
H1=[P1,-P1,-P1,O_p;-P1,P1,P1,O_p;-P1,P1,P1,O_p;O_p,O_p,O_p,O_p];
P2=kernelfunction(kerneltype, A2, A2, kernelparamA);
H2=[P2,-P2,O_p,P2;-P2,P2,O_p,-P2;O_p,O_p,O_p,O_p;P2,-P2,O_p,P2];
H=H1+H2;

% solve the first QPP - E
B2_A2=kernelfunction(kerneltype, B2, A2, kernelparamA);
B1_A1=kernelfunction(kerneltype, B1, A1, kernelparamA);
linear_obj_p = CA1*ones(l_n,1)'*[B2_A2-B1_A1, -B2_A2 + B1_A1, B1_A1, B2_A2];

% ineq_cons
A = [E_p E_p zeros(l_p, l_p) zeros(l_p, l_p)];
B = D1 * ones(l_p, 1);

% eq_cons
Aeq = [];
Beq = [];
LB = zeros(2 * (l_n + l_p), 1);
UB = [zeros(2 * l_p, 1); ones(l_n, 1) * CA1; ones(l_n, 1) * CB1 ];

[ pai ] = quadprog(H, linear_obj_p, A, B, Aeq, Beq, LB, UB, [], global_options);
pai = round(pai, num_truncation);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solve the second QPP - F
P1=kernelfunction(kerneltype, B1, B1, kernelparamA);
H1=[P1,-P1, P1,O_n;-P1,P1,-P1,O_n;P1,-P1,P1,O_n;O_n,O_n,O_n,O_n];
P2=kernelfunction(kerneltype, B2, B2, kernelparamA);
H2=[P2,-P2,O_n,P2;-P2,P2,O_n,-P2;O_n,O_n,O_n,O_n;P2,-P2,O_n,P2];
H=H1+H2;
A1_B1=kernelfunction(kerneltype, A1, B1, kernelparamA);
A2_B2=kernelfunction(kerneltype, A2, B2, kernelparamA);
linear_obj_p = CB1*ones(l_p,1)'*[A1_B1+A2_B2, -A1_B1 - A2_B2, A1_B1, A2_B2];     % obtain linear vector

% ineq_cons
A = [zeros(l_n, l_n) zeros(l_n, l_n) E_n E_n];
B = D2 * ones(l_n, 1);

% eq_cons
Aeq = [];
Beq = [];

LB = zeros(2 * (l_n + l_p), 1);
UB = [ones(l_p, 1) * CA2; ones(l_p, 1) * CB2; D2 * ones(2 * l_n, 1)];

[ fai ] = quadprog(H, linear_obj_p, A, B, Aeq, Beq, LB, UB, [], global_options);
fai = round(fai, num_truncation);

% this is the key to generate decision hyper-plane



%% Predict
testXA=test_set1(:,1:end-1);
testXB=test_set2(:,1:end-1);
testy=test_set1(:,end);

kermatA = kernelfunction(kerneltype, testXA, A1, kernelparamA);
kermatAA = kernelfunction(kerneltype, testXA, B1, kernelparamA);
kermatB = kernelfunction(kerneltype, testXB, A2, kernelparamA);
kermatBB = kernelfunction(kerneltype, testXB, B2, kernelparamA);
% kermatA=testXA;
% kermatB=testXB;
% viewA's origin result
origin_viewAp = [-kermatA, kermatA, kermatA, zeros(size(kermatA))] * pai - CA1*kermatAA*ones(l_n,1);
origin_viewAn = CB1*kermatA*ones(l_p,1) - [-kermatAA, kermatAA, -kermatAA, zeros(size(kermatAA))]* fai;

origin_viewBp = [kermatB, -kermatB, zeros(size(kermatB)), kermatB] * pai - CA1*kermatBB*ones(l_n,1);
origin_viewBn = CB1*kermatB*ones(l_p,1) - [-kermatBB, kermatBB, -kermatBB, zeros(size(kermatBB))]* fai;


if length(origin_viewAp) == length(origin_viewBp)   % if the sample have two view, then do two view decision
    origin_resultA = abs(origin_viewAn) - abs(origin_viewAp);
    if isnan(origin_resultA) == 1
        origin_resultA = zeros(size(origin_viewAn));
    end

    origin_resultB = abs(origin_viewBn) - abs(origin_viewBp);
    if isnan(origin_resultB) == 1
        origin_resultB = zeros(size(origin_viewBn));
    end
    origin_result = origin_resultA + origin_resultB;
    bina_result = sign(origin_result);
    bina_A = sign(origin_resultA);
    bina_B = sign(origin_resultB);
else
    origin_resultA = abs(origin_viewAn) - abs(origin_viewAp);
    origin_resultB = abs(origin_viewBn) - abs(origin_viewBp);
    bina_result = [];                   % else, do not
    bina_A = sign(origin_resultA);
    bina_B = sign(origin_resultB);
end

[accuracy, recall,precision,F_1score,G_means] = judgement(bina_result, testy);
[accuracy_viewA, recall_A,precision_A,F_1score_A,G_means_A] = judgement(bina_A, testy);
[accuracy_viewB, recall_B,precision_B,F_1score_B,G_means_B] = judgement(bina_B, testy);

time =toc;


end