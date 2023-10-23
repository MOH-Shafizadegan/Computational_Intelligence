clear; clc; close all;

%                          define data
% class with label 1
class1 = [0.1, 0.7; 0.28, 0.58; 0.45, 0.15; 0.6, 0.3];
% class with label -1
class2 = [0.12, 1; 0.35, 0.98; 0.7, 0.65; 0.95, 0.45];

% scatter plot data
figure;
scatter(class1(:,1), class1(:,2)); hold on;
scatter(class2(:,1), class2(:,2), 'filled');
xlabel('x1');
ylabel('x2');
xlim([0, 1.2]);
ylim([0, 1.2]);
legend('class 1', 'class -1')

% train the model
n_iter = 100;
eta = 0.5;
init_w = [1; -1];
init_theta = -0.2;

% a matrix to record the error for each itteration
error = zeros(length(class1)+length(class2), n_iter);

% a matrix to save the data used for updating in each itteration
selected_data = zeros(n_iter,2);

x = [class1 ; class2];
% all data labels within an array (first class 1 then class -1)
out = [1 1 1 1 -1 -1 -1 -1];

w = init_w;
theta = init_theta;
i = 1;
for i=1:n_iter
    y = -1*ones(1,length(x));
    y(w' * x' >= theta) = 1;
    error(:,i) = out - y;
    error_labeled = nonzeros(error(:,i)); % get the nonzero elements
    % stop the process if there is no error :)
    if isempty(error_labeled)
        break;
    else
        % randomly choose one of the data labeled wrongly
        idx = randi(length(error_labeled)); % get a random index
        idx = find(error(:,i) == error_labeled(idx), 1);
        % save the chosen data
        selected_data(i, :) = x(idx, :);
        % update weights and threshold
        w = w + eta * (out(idx) - y(idx)) * x(idx, :)';
        theta = theta - eta * (out(idx) - y(idx));
        fprintf('number of iterations: %d \n', i)
        fprintf('the resulting weigths: w1=%d , w2=%d \n', w(1), w(2));
        fprintf('threshold = %d \n', theta);
        disp('-------------------------------')
    end
end