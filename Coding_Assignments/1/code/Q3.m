clc; clear; close all;

% load dataset
irisTable = readtable('../iris.csv');
irisData = table2array(irisTable(:,1:4));

% Extract data for Iris-setosa and Iris-versicolor classes
setosa_idx = find(strcmp(irisTable.Var5, 'Iris-setosa'));
versicolor_idx = find(strcmp(irisTable.Var5, 'Iris-versicolor'));
virginica_idx = find(strcmp(irisTable.Var5, 'Iris-virginica'));

setosa_data = irisData(setosa_idx, :);
versicolor_data = irisData(versicolor_idx, :);
virginica_data = irisData(virginica_idx, :);

%%
clc; close all;

figure;
subplot(2,2,1);
scatter_features(setosa_data, versicolor_data, virginica_data, 1, 2, 3);
subplot(2,2,2);
scatter_features(setosa_data, versicolor_data, virginica_data, 1, 2, 4);
subplot(2,2,3);
scatter_features(setosa_data, versicolor_data, virginica_data, 2, 3, 4);
subplot(2,2,4);
scatter_features(setosa_data, versicolor_data, virginica_data, 1, 3, 4);

% => use features 1 , 3 , 4

%%  train a TLU for separating class 1 and class 2

clc; close all;

selected_features = [1, 3, 4];

P = randperm (50);
train_data = [setosa_data(P(1:40), selected_features) ; ...
              versicolor_data(P(1:40), selected_features)];

% all data labels within an array (first class 1 then class 0)
out = [ones(1,40), zeros(1,40)];

n_iter = 100;
eta = 0.5;

[w_12, theta_12, iter_num] = online_learning([0; 0; 0], 0, eta, n_iter, train_data, out);

disp('Separating class 1 and 2');
fprintf('number of iterations: %d \n', iter_num)
fprintf('the resulting weigths: w1=%d , w2=%d , w3=%d\n', w_12(1, end), ...
        w_12(2, end), w_12(3, end));
fprintf('threshold = %d \n', theta_12(end));

figure;
scatter_features(setosa_data, versicolor_data, virginica_data, 1, 3, 4);
hold on;
plot_plane(w_12(1, end), w_12(2, end), w_12(2, end), theta_12(end))

%% test accuracy for class 1 and 2

test_data = [setosa_data(P(41:50), selected_features) ; ...
              versicolor_data(P(41:50), selected_features)];
          
out_test = [ones(1,10), zeros(1,10)];

y_train_12 = w_12(:, end)' * train_data' >= theta_12(end); 
y_test_12 = w_12(:, end)' * test_data' >= theta_12(end);

acc_12_test = sum(y_test_12 == out_test)/length(out_test) * 100;
acc_12_train = sum(y_train_12 == out)/length(out) * 100;

fprintf('Accuracy of classification of class 1 and 2 on test data: %d  \n', acc_12_test);
fprintf('Accuracy of classification of class 1 and 2 on train data: %d  \n', acc_12_train);

%% train a TLU for separating class 2 and class 3

P = randperm (50);
train_data = [versicolor_data(P(1:40), selected_features) ; ...
              virginica_data(P(1:40), selected_features)];

% all data labels within an array (first class 1 then class 0)
out = [ones(1,40), zeros(1,40)];

eta = 0.01;
n_iter = 1000;

[w_23, theta_23, n_iter_23] = online_learning([0; 0; 0], 0, eta, n_iter, train_data, out);

disp('Separating class 2 and 3');
fprintf('number of iterations: %d \n', n_iter_23)
fprintf('the resulting weigths: w1=%d , w2=%d , w3=%d\n', w_23(1, end), ...
        w_23(2, end), w_23(3, end));
fprintf('threshold = %d \n', theta_23(end));

figure;
scatter_features(setosa_data, versicolor_data, virginica_data, 1, 3, 4);
hold on;
plot_plane(w_23(1, end), w_23(2, end), w_23(2, end), theta_23(end))

%% test accuracy for class 2 and 3

test_data = [versicolor_data(P(41:50), selected_features) ; ...
              virginica_data(P(41:50), selected_features)];
          
out_test = [ones(1,10), zeros(1,10)];

y_train_23 = w_23(:, end)' * train_data' >= theta_23(end); 
y_test_23 = w_23(:, end)' * test_data' >= theta_23(end);

acc_23_test = sum(y_test_23 == out_test)/length(out_test) * 100;
acc_23_train = sum(y_train_23 == out)/length(out) * 100;

fprintf('Accuracy of classification of class 2 and 3 on test data: %d  \n', acc_23_test);
fprintf('Accuracy of classification of class 2 and 3 on train data: %d  \n', acc_23_train);

%%

figure;
scatter_features(setosa_data, versicolor_data, virginica_data, 1, 3, 4);
hold on;
plot_plane(w_12(1, end), w_12(2, end), w_12(2, end), theta_12(end))
plot_plane(w_23(1, end), w_23(2, end), w_23(2, end), theta_23(end))


%% 

function scatter_features(data1, data2, data3, f1, f2, f3)
    scatter3(data1(:,f1), data1(:,f2), data1(:,f3), 'filled', 'MarkerFaceColor', 'r');
    hold on;
    scatter3(data2(:,f1), data2(:,f2), data2(:,f3), 'filled', 'MarkerFaceColor', 'g');
    scatter3(data3(:,f1), data3(:,f2), data3(:,f3), 'filled', 'MarkerFaceColor', 'b');
    xlabel(sprintf('feature %d', f1));
    ylabel(sprintf('feature %d', f2));
    zlabel(sprintf('feature %d', f3));
    title(sprintf('feature %d vs. feature %d vs. feature %d', f1, f2, f3));
    legend('Iris-setosa', 'Iris-versicolor', 'Iris-virginica');
    hold off;
end

function [w_array, theta_array, iter_num] = online_learning(init_w, init_theta, eta, n_iter, x, out)
    
    w = init_w;
    theta = init_theta;

    w_array = [w];
    theta_array = [theta];

    for i=1:n_iter
        e = 0;
        for j=1:length(x)
            y = w' * x(j,:)' >= theta;
            if y ~= out(j)
                theta = theta - eta * (out(j)-y);
                theta_array = [theta_array theta];
                w = w + eta * (out(j)-y) * x(j,:)';
                w_array = [w_array w];
                e = e + abs(out(j)-y);
            end
        end
        if e <= 0
            break;
        end
    end
    iter_num = i;
end

function [w_array, theta_array, iter_num] = batch_learning(init_w, init_theta, eta, n_iter, x, out)
    
    w = init_w;
    theta = init_theta;

    w_array = [w];
    theta_array = [theta];

    for i=1:n_iter
        e = 0;
        theta_c = 0;
        w_c = zeros(2,1);
        for j=1:length(x)
            y = w' * x(j,:)' >= theta;
            if y ~= out(j)
                theta_c = theta_c - eta * (out(j)-y);
                w_c = w_c + eta * (out(j)-y) * x(j,:)';
                e = e + abs(out(j)-y);
            end
        end
        theta = theta + theta_c;
        theta_array = [theta_array theta];
        w = w + w_c;
        w_array = [w_array w];
        if e <= 0
            break;
        end
    end
    iter_num = i;
end

function plot_plane(w1, w2, w3, theta)
    % Inputs: w1, w2, w3 are the weights of the plane equation: w1*x + w2*y + w3*z = theta
    
    % Create a grid of x and y values
    [x, y] = meshgrid(0:10, 0:10);

    % Solve for z values using the plane equation
    z = (theta - w1*x - w2*y) / w3;

    % Plot the plane using surf function
    surf(x, y, z);
end