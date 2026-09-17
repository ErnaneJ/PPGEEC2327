function [grad,res]=gradientnn(M,x,t,ww,layer_size,obj_option,act_option, lambda);
% output:
%  res - objective function value
%  grad - (objective function respective to weights)
%
% Input:
% M   - # data examples
%(x,t)- Training data pairs
%ww- filter for all layers
% t   - # observed data
% layer_size - indicate nn structure
%obj_option=1 for L2, 2 for likelihood
%act_option=1 for sigmoid, 2 for ReLU

% Part 1: Feedforward the neural network and return the cost in the
%         variable res.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients.
%         You should return the partial derivatives of the cost function with
%         respect weight, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K.

%% Setup some useful variables
layer_num=numel(layer_size); %layer number include the input and output
aa{1}=x; %first layer is the input layer
penalize=0;
%% Part 1: Feedforward the neural network and return the cost in the variable res.

% Do forward propagation
for iter =1:layer_num-1 %%%----- N (in book) ------%%%
    % ones(1, m) is for bias
    aa{iter} = [ones(1, M); aa{iter}];
    %%%----- z[n]=W[n]a[n-1]  ------%%%
    zz{iter}=ww{iter}*aa{iter};
    %%%----- a[n]=g(z[n])------%%%
    if iter ==layer_num-1 %output layer has to use sigmoid
        [aa{iter+1},~] = activation(zz{iter},1);
    else
        [aa{iter+1},~] = activation(zz{iter},act_option);
    end
    %Add some regularization
    penalize =penalize+ sum(sum(ww{iter}.^ 2)); % include regularization for bias
end
% final output for forward propagation Ypred
t_pred = aa{layer_num};

% calculate the objective function and the dirivative of objective function with respect to t_pred using likelihood or L2 type
%obj_option=1 for L2, 2 for likelihood
[res,in]= misfit( t_pred,t,1/M,obj_option);

% add regularization
res = res + (lambda/(2*M)) * penalize; 
%% Part 2: Implement the backpropagation algorithm to compute the gradients.
%  writen according to Backpropagation Operation 

% Implement backpropagation
for iter=layer_num-1:-1:1
    %'sigmoidgrad' means compute the gradient of the sigmoid function
    if iter ==layer_num-1 %output layer has to use sigmoid
        [~,dg]=activation(zz{iter},1);
    else
        [~,dg]=activation(zz{iter},act_option);  %%%----- in=dg[i].*in (in book) ------%%%
    end
    in=dg.*in;
    % in is the backward field, A is the forward field
    grad{iter}=in*aa{iter}'+ (lambda/M) * ww{iter}; %%%----- de(j,k)=in*(a[i-2])T (in book) ------%%%
    % update in for calculation of the gradient with respect to next weights
    in=ww{iter}'*in;  %%%----- in=W'[i]*in (in book) ------%%%
    in = in(2:end, :);
end

end