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

X=[ones(m,1) X];
temp1=X*(Theta1');
temp1=sigmoid(temp1);
temp1=[ones(size(temp1,1),1) temp1];
temp2=temp1*(Theta2');
h0=sigmoid(temp2);

part1=0;
for i=1:num_labels
	why=(y==i);
	hui=h0(:,i);
	part1=part1+(log(hui')*why)+(log(1-hui')*(1-why));
end
part2=sum(sum(Theta1.^2));
part2=part2-sum(Theta1(:,1).^2); %Removing the first column
part3=sum(sum(Theta2.^2));
part3=part3-sum(Theta2(:,1).^2);  %Removing the first column
jedi=(part2+part3)*(lambda);
jedi=jedi/2;
J=jedi-part1;
J=J/m;

tri1=0;
tri2=0;
%Screwed up code follows.Rewrite.
for i=1:m
	%Step 1
	ex=X(i,:);
	z2=ex*(Theta1');
	a2=sigmoid(z2);
	a2=[ones(size(a2,1),1) a2];
	z3=a2*(Theta2');
	a3=sigmoid(z3);
	%Step 2
	for j=1:num_labels
		del3(j)=(a3')(i)-(y(i)==j);
	end
	%Step 3
	death_star=(Theta2')*(del3');
	death_star=death_star(2:end);
	del2=death_star.*(sigmoidGradient(z2'));
	del2=del2(2:end);
	%Step 4
	tri2=tri2+((a2')*(del3));
	tri1=tri1+((ex')*(del2'));	
end

Theta1_grad=tri1./m;
Theta2_grad=tri2./m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
