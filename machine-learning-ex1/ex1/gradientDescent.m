function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    J=0;
    K=0;

    for i = 1:m
        h0 = (theta')*(X(i,:)');
        h0 = h0 - y(i,:);
        h0=h0*X(i,1);
        J=J+(h0);
    endfor
    temp1=(J*(alpha))/m;

    for i = 1:m
        h1 = (theta')*(X(i,:)');
        h1 = h1 - y(i,:);
        h1=h1*X(i,2);
        K=K+(h1);
    endfor

    temp2=(K*(alpha))/m;
    temp3=[temp1;temp2];
    theta=theta-temp3;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
