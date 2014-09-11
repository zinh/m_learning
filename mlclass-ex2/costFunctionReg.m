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

% normal function
for i = 1:m
  row = X(i,:);
  h = row * theta;
  s = sigmoid(h);
  J = J + (-y(i) * log(s) - (1 - y(i)) * log(1 - s) );
end

p = 0
% regular parameter
for i = 2:size(theta)
  p = p + theta(i)^2;
end
p = p * lambda / (2*m);
J = J / m + p;


s = 0;
i = 1;
for r = 1:m
  row = X(r,:);
  h = row * theta;
  sig = sigmoid(h);
  s = s + (sig - y(r))*X(r, i);
end
grad(i) = s / m;

for i = 2:size(grad)
  s = 0;
  for r = 1:m
    row = X(r,:);
    h = row * theta;
    sig = sigmoid(h);
    s = s + (sig - y(r))*X(r, i);
  end
  p = lambda * theta(i) / m;
  grad(i) = s / m + p;
end

% =============================================================

end
