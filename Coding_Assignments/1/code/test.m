clc; clear; close all;

% load dataset
irisTable = readtable('../iris.csv');
irisData = table2array(irisTable(:,1:4));

% Extract data for Iris-setosa and Iris-versicolor classes
setosa_idx = find(strcmp(irisTable.Var5, 'Iris-setosa'));
versicolor_idx = find(strcmp(irisTable.Var5, 'Iris-versicolor'));

setosa_data = irisData(setosa_idx, :);
versicolor_data = irisData(versicolor_idx, :);

%%
clc; close all;

P = randperm (50);

train_data = [setosa_data(P(1:5), 3:4) ; versicolor_data(P(1:5), 3:4)];

% all data labels within an array (first class 1 then class 0)
out = [ones(1,5), zeros(1,5)];

% online learning
% train the model
n_iter = 100;
eta = 0.5;

[w_online, theta_online, iter_num] = batch_learning(eta, n_iter, train_data, out);

disp('Online learning');
fprintf('number of iterations: %d \n', iter_num)
fprintf('the resulting weigths: w1=%d , w2=%d \n', w_online(1, end), w_online(2, end));
fprintf('threshold = %d \n', theta_online(end));

figure;
scatter_features(setosa_data(P(1:5),:), versicolor_data(P(1:5),:), 3,4); hold on;
title('Online learning')
plot_line(w_online(1, end), w_online(2, end), theta_online(end)); hold off;


%% functions

function scatter_features(data1, data2, f1,f2)
    data1
    data2
    % Plotting features: sepal length vs sepal width
    scatter(data1(:,f1), data1(:,f2), 'filled', 'MarkerFaceColor', 'r');
    hold on;
    scatter(data2(:,f1), data2(:,f2), 'filled', 'MarkerFaceColor', 'g');
    xlabel(sprintf('feature %d', f1));
    ylabel(sprintf('feature %d', f2));
    title(sprintf('feature %d vs. feature %d', f1, f2));
    legend('Iris-setosa', 'Iris-versicolor');
    hold off;
end

function [w_array, theta_array, iter_num] = online_learning(eta, n_iter, x, out)
    
    init_w = [0; 0];
    init_theta = 5;

    w = init_w;
    theta = init_theta;

    w_array = [w];
    theta_array = [theta];

    for i = 1:n_iter
        e = 0;
        for j = 1:length(x)
            % Compute the output of the threshold logic unit (TLU)
            y = w' * x(j,:)' >= theta;

            % Check if the output is different from the target
            if y ~= out(j)
                % Update the threshold and store its value
                theta = theta - eta * (out(j) - y);
                theta_array = [theta_array theta];

                % Update the weights and store their values
                w = w + eta * (out(j) - y) * x(j,:)';
                w_array = [w_array w];

                % Update the error count
                e = e + abs(out(j) - y);
            end
        end

        % Check if the error count is less than or equal to zero
        % If true, exit the loop as the training is complete
        if e <= 0
            break;
        end
    end
    iter_num = i;
end

function plot_line(w1, w2, theta)
    slope = -w1/w2;
    intercept = theta/w2;

    x = 0:0.1:5;
    y = slope*x + intercept;

    plot(x,y,'r'); xlabel('x'); ylabel('y');
end

function [w_array, theta_array, iter_num] = batch_learning(eta, n_iter, x, out)
    
    init_w = [-1; -1];
    init_theta = -5;

    w = init_w;
    theta = init_theta;

    w_array = [w];
    theta_array = [theta];

    for i = 1:n_iter
        e = 0;
        theta_c = 0;
        w_c = zeros(2,1);

        for j = 1:length(x)
            % Compute the output of the threshold logic unit (TLU)
            y = w' * x(j,:)' >= theta;

            % Check if the output is different from the target
            if y ~= out(j)
                % Update the temporary threshold and weights
                theta_c = theta_c - eta * (out(j) - y);
                w_c = w_c + eta * (out(j) - y) * x(j,:)';

                % Update the error count
                e = e + abs(out(j) - y);
            end
            fprintf('wx = %d \n', w' * x(j,:)')
            fprintf('y = %d \n', y)
            fprintf('e = %d \n', abs(out(j) - y))
            fprintf('detatheta = %d \n', - eta * (out(j) - y))
            fprintf('delta w = %d \n', eta * (out(j) - y) * x(j,:)')
            disp('-------')
        end

        % Update the threshold and weights using the temporary values
        theta = theta + theta_c;
        theta_array = [theta_array theta];
        w = w + w_c;
        w_array = [w_array w];
        % Check if the error count is less than or equal to zero
        % If true, exit the loop as the training is complete
        if e <= 0
            break;
        end
    end
    iter_num = i;
end

