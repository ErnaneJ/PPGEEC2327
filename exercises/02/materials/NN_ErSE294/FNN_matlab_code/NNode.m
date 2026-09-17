%% Simple Neural Network LAB 
% Learn how to to do forward and backward propogation according to chsingle.pdf by Gerard Schuster
% coded by Zongcai Feng and Gerard Schuster

% This is the main program to run Neural Network
% It involves function
%  gradientnn.m - Neural network cost function and gradient calculation
%  misfit.m - Objective function and gradient with repected to predicted output
%              (currently have likelihood and L2)
%  activation.m - activation function and gradient with repected to Z (currently use sigmoid and ReLU)
%  Displaynn.m - predict the output and plot

%% ================= Set up input =================
clear ; close all; clc

% x(N,M)  - input- M input feature vectors with size Nx1
M=100;  % # of equations constraint
N=5;    % # of unknowns (w0, w1, wN-1)
x=zeros(N,M);

for i=1:M
    x(:,i)=round(rand(N,1));
end
% ---- to balance the number of 0 examples and 1 exmaples -----
is0_token=0;
for i=1:M
    if sum(x(:,i))==0
        is0_token=is0_token+1;
    end
end
x=[zeros(N,M-is0_token),x];
M=size(x,2);
rank = randperm(M);
x=x(:,rank);
% ----------------giving labels-----------------------------------------------
t=zeros(M,1);tp=t;
for i=1:M
    if sum(x(:,i))==0
        t(i)=1;
    end
end

% Randomly select 100 data points to display
sel = randperm(M);
sel = sel(1:100);
displayData(x(:, sel)',N);
%% ================= Set up parameters for NN structure=================--------------------------------------------------------------------------------
display('-----------------------------------------------------------Instruction of NN structure---------------------------------------------------------------------------------------');
display('The NN code requires the input the number of nodes in each layer.');
display('(The layers defines in this NN structure do not include the layers for input feathuires but include the layers for output labels)');
display('(E.g:[15,10,5,1], three hidden layers and their nodes numbers are 15,10,5, respectively, 1 is the ouptlayer, consistent with variable)');
display('-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ');
layer_temp=input('Please input the number of nodes in each layer:  ');
%layer_temp=[15,10,5,1];
display(' ');
obj_option=input('Please input the objective function type, 1 for L2 norm, 2 for likelihood:   ');
display(' ');
act_option=input('Please input the active function type, 1 for sigmoid norm, 2 for ReLU:   ');

layer_size=[N,layer_temp]; % layer_size include the input and output layer
layer_num=numel(layer_size); %layer number include the input and output
%% ================ Initializing Pameters for Training ================
%  initialize the weights of the neural network with random number
for ilayer=1:layer_num-1
    ww{ilayer}=rand(layer_size(ilayer+1),layer_size(ilayer)+1)*0.1;
end
alpha=1;         % step size
nit=10000;    % total interation
res=zeros(nit,1);  %used to record the objetive function value in every iterations
kk=0;
lambda = 0.01; %for regularization, if add regularization, suggest lambda<0.01
%% =================== Training NN ===================
%  To train your neural network, we will now use steepest decent and line search
fprintf('\nTraining Neural Network... \n')

for k=2:nit     % Looping over iterations
    alpha=1;  
    %gradientnn ouput: grad is the gradient and res(k) is the objetive function value
    [grad,res(k)]=gradientnn(M,x,t',ww,layer_size,obj_option,act_option,lambda);
    %update the weights
    for ilayer=1:layer_num-1
        ww{ilayer}=ww{ilayer}-alpha*grad{ilayer};
    end
    %---------------line search-----------------------------------------------------------
    [~,res1]=gradientnn(M,x,t',ww,layer_size,obj_option,act_option,lambda);
    while (res1>res(k)) && (alpha>0.00001)
        alpha=alpha*0.5;
        for ilayer=1:layer_num-1
            ww{ilayer}=ww{ilayer}-alpha*grad{ilayer};
        end
        [~,res1]=gradientnn(M,x,t',ww,layer_size,obj_option,act_option,lambda);
    end
    %-------------------------------------------------------------------------------
    %kk=kk+1
end
%% ================= Implement Predict =================
%  After training the neural network, you will now implement the "Displaynn" function to use the
%  neural network to predict the labels of the training set.
figure(4);Displaynn(M,res,nit,ww,layer_size,x,t,act_option);pause(0.5)
