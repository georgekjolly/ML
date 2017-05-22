function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


%fprintf( 'size of X : raw = %d and column = %d \n', size(X,1), size(X,2));
%X
%fprintf( 'size of y : raw = %d and column = %d \n', size(y,1), size(y,2));
%y
%fprintf( 'size of theta : raw = %d and column = %d \n', size(theta,1), size(theta,2));
%theta


J_Cost = (1/(2*m))*(sum(((X*theta) - y).^2));

theta_tmp = theta;
theta_tmp(1:1) = 0;

J_Reg = (lambda/(2*m))*sum(theta(2:end) .^ 2);

J = J_Cost + J_Reg;


grad = (((X*theta - y)' * X)./m)' + ((lambda/m) * [0; theta(2:end)]);

%fprintf( 'size of grad : raw = %d and column = %d \n', size(grad',1), size(grad',2));

% =========================================================================

grad = grad(:);

end
