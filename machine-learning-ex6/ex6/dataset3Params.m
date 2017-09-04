function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
t = [0.01 0.03 0.1 0.3 1 3 10 30];
%t = [1 3];
re = zeros(length(t),3);
k = 0;
for i = 1:length(t)
  for j = 1:length(t)
    C = t(i);
    sigma = t(j);
    % fprintf("C = ");disp(C);
    % fprintf("\nsigma = \n");disp(sigma);
    k = k + 1;
    % fprintf("k = %d\n",k);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    er = mean(double(predictions ~= yval));
    re(k,:) = [C sigma er];
    % fprintf("result = \n");
    % disp(re(k,:));
  end;
end;
[x ix] = min(re'(3,:));
%[C sigma er] = re(ix);
disp(re);
C = re(ix,1);
sigma = re(ix,2);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
