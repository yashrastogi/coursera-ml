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

a1 = transpose([ones(m,1) X]);
z2 = Theta1 * a1;
a2 = sigmoid(z2);
a2 = [ones(1, m); a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);
prediction = a3;
temp = y;
y = zeros(m, num_labels);
for i=1:m,
    for j=1:num_labels,
        if temp(i) == j,
            y(i, j) = 1;
        end
    end
end

summ=0;
y = transpose(y);
for i=1:m,
    for k=1:num_labels,
        summ = summ - ( y(k, i) * log(prediction(k, i)) + (1 - y(k, i)) * log(1 - prediction(k, i)));
    end
end
J = summ / m;

% keyboard;

% Regularization for 3 layer neural network architecture

% Regularization parameter: Layer 1 (input layer)
theta1Rows = size(Theta1)(1,1);
theta1Columns = size(Theta1)(1,2);
paramL1=0;
for j=1:theta1Rows,
    for k=2:theta1Columns,
        paramL1 = paramL1 + Theta1(j, k) ^ 2;
    end
end

% Regularization parameter: Layer 2 (hidden layer)
theta2Rows = size(Theta2)(1,1);
theta2Columns = size(Theta2)(1,2);
paramL2=0;
for j=1:theta2Rows,
    for k=2:theta2Columns,
        paramL2 = paramL2 + Theta2(j, k) ^ 2;
    end
end

% Total Regularization Term
regTerm = lambda * ( paramL1+paramL2 ) / (2*m);

J = J + regTerm;


% Gradient calculation
Delta3 = a3 - y;
z2 = [ones(1, m); z2];
Delta2 = transpose(Theta2) * Delta3 .* sigmoidGradient(z2);
Delta2 = Delta2(2:size(Delta2)(1,1), 1:size(Delta2)(1,2));

Theta2_grad = Theta2_grad + Delta3 * transpose(a2);
Theta2_grad = Theta2_grad / m;
temp = lambda * Theta2 / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + temp(:, 2:end) ;
Theta1_grad = Theta1_grad + Delta2 * transpose(a1);
Theta1_grad = Theta1_grad / m; 
temp = lambda * Theta1 / m;
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + temp(:, 2:end) ;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
