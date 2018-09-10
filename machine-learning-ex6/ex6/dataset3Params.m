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
C_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_arr = [0.01 0.03 0.1 0.3 1 3 10 30];
results = zeros(64,3);
i = 0;

for temp_C=C_arr
    for temp_sigma=sigma_arr
        i = i+1;
        model= svmTrain(X, y, temp_C, @(x1, x2) gaussianKernel(x1, x2, temp_sigma)); 
        predictions = svmPredict(model,Xval);
        error = mean(double(predictions ~= yval));
        results(i,:) = [temp_C temp_sigma error];
    end
end

sorted_results = sortrows(results, 3); % sort matrix by column #3, the error, ascending

C = sorted_results(1,1);
sigma = sorted_results(1,2);
% =========================================================================

end
