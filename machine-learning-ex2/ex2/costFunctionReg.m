function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

h0 = theta' * (X');
lol=sigmoid(h0);
J = ((-log(lol)*y)-(log(1-lol)*(1-y)))+((lambda/2)*((sum(theta.^2))-(theta(1)^2))); %Don't consider theta(0) in the lambda sum
J = J/m;

lol=lol';
grad = (1/m)*(sum(lol-y,2)')*X + (lambda/m)*(theta');
grad(1) = ((1/m)*(sum(lol-y,2)')*X)(1);
grad=grad';
% =============================================================

end
