function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X_input = [ones(size(X,1),1)  X];
a2 = sigmoid(Theta1 * X_input'); % a2 is 25x5000
% make it 26x5000
a2_t = a2';
a2_input = [ones(size(a2_t,1),1)  a2_t];
a3 = sigmoid(Theta2 * a2_input'); % a3 is 10x5000
[maxval, maxindices] = max(a3);
p = maxindices';






% =========================================================================


end
