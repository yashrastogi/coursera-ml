function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

x = size(z)(:,1);
y = size(z)(:,2);

for i=1:x,
    for j=1:y,
        g(i, j) = 1/(1+e^(-z(i, j)));
end


% =============================================================

end
