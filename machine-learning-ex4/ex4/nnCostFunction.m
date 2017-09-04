function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% fprintf("chenzehuifengle\n")
% disp(size(Theta1));
% disp(size(Theta2));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
J1 = 0;
J2 = sum(nn_params .^ 2) - sum(Theta1(:,1) .^ 2) - sum(Theta2(:,1) .^ 2);
h = [ones(size(X,1),1) X] * (Theta1)';
h = sigmoid(h);
h = [ones(size(h,1),1) h] * (Theta2)';
h = sigmoid(h);
for i = 1:m
  y1 = zeros(num_labels,1);
  y1(y(i)) = 1;
  J1 = J1 + sum(y1 .* log(h(i,:)') + (1 - y1) .* log(1 - h(i,:)'));
end;
J = (-1/m) * J1 + (lambda / (2 * m)) * J2;


%init
w1 = zeros(size(Theta1));
w2 = zeros(size(Theta2));
x = [ones(m,1) X];
for i = 1:m
  y1 = zeros(1,num_labels);
  y1(y(i)) = 1;
  %FP
  a1 = x(i,:);
  z2 = a1 * (Theta1)';
  a2 = sigmoid(z2);
  a2 = [1,a2];
  z3 = a2 * (Theta2)';
  a3 = sigmoid(z3);
  
  %Cal p
  p3 = a3 - y1;
  p2 = ((Theta2)' * p3')' .* (a2 .* (1 .- a2));
  
  %Cal w
  w2 = w2 + p3' * a2;
  w1 = w1 + p2(2:end)' * a1;
  
end;

Theta1_grad = (1 / m) * w1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda / m) * (Theta1)(:,2:end);
Theta2_grad = (1 / m) * w2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda / m) * (Theta2)(:,2:end);
% disp(size(Theta1_grad));
% disp(size(Theta2_grad));
  

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
