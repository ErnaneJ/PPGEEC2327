function [J, grad] = costFunction2(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%%% Compute the cost function
for ix=1:m
    J=J+( -y(ix)*log(1/(1+exp(-(X(ix,:)*theta)))) -(1-y(ix))*log(1-(1/(1+exp(-(X(ix,:)*theta))))));    
end
    J=J/m;
%%% Compute the gradient
for ix=1:m
   grad_temp=grad;
   grad_temp(1)=grad(1)+(1/(1+exp(-(X(ix,:)*theta)))-y(ix))*X(ix,1);
   grad_temp(2)=grad(2)+(1/(1+exp(-(X(ix,:)*theta)))-y(ix))*X(ix,2);
   grad_temp(3)=grad(3)+(1/(1+exp(-(X(ix,:)*theta)))-y(ix))*X(ix,3);
   grad_temp(4)=grad(4)+(1/(1+exp(-(X(ix,:)*theta)))-y(ix))*X(ix,4);
   grad=grad_temp;
end

grad=grad./m;

% =============================================================

end
