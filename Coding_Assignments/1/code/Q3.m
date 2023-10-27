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


%%
clc; close all;

P = randperm (50);
train_data = [setosa_data(P(1:40), 3:4) ; versicolor_data(P(1:40), 3:4)];

% all data labels within an array (first class 1 then class 0)
out = [ones(1,40), zeros(1,40)];

% online learning
% train the model
n_iter = 100;
eta = 0.5;

[w_online, theta_online, iter_num] = online_learning(eta, n_iter, train_data, out);

disp('Online learning');
fprintf('number of iterations: %d \n', iter_num)
fprintf('the resulting weigths: w1=%d , w2=%d \n', w_online(1, end), w_online(2, end));
fprintf('threshold = %d \n', theta_online(end));

figure;
scatter_features(setosa_data, versicolor_data, 3,4); hold on;
title('Online learning')
plot_line(w_online(1, end), w_online(2, end), theta_online(end)); hold off;

%% batch learning

n_iter = 100;
eta = 0.5;

[w_batch, theta_batch, n_iter_batch] = batch_learning(eta, n_iter, train_data, out);

disp('Batch learning');
fprintf('number of iterations: %d \n', n_iter_batch)
fprintf('the resulting weigths: w1=%d , w2=%d \n', w_batch(1, end), w_batch(2, end));
fprintf('threshold = %d \n', theta_batch(end));

figure;
scatter_features(setosa_data, versicolor_data, 3,4); hold on;
title('Batch learning')
plot_line(w_batch(1, end), w_batch(2, end), theta_batch(end)); hold off;


%% 

% online learning w and theta plots
figure;
subplot(3,1,1)
plot(w_online(1,:))
title('Online learning')
xlabel('iteration')
ylabel('w1');
subplot(3,1,2)
plot(w_online(2,:))
xlabel('iteration')
ylabel('w2');
subplot(3,1,3)
plot(theta_online)
xlabel('iteration')
ylabel('theta');

% batch learning w and theta plots
figure;
subplot(3,1,1)
plot(w_batch(1,:))
title('Batch learning')
xlabel('epoch')
ylabel('w1');
subplot(3,1,2)
plot(w_batch(2,:))
xlabel('epoch')
ylabel('w2');
subplot(3,1,3)
plot(theta_batch)
xlabel('epoch')
ylabel('theta');

%%

test_data = [setosa_data(P(41:50), 3:4) ; versicolor_data(P(41:50), 3:4)];
out_test = [ones(1,10), zeros(1,10)];

y_online = w_online(:, end)' * test_data' >= theta_online(end);

y_batch = w_batch(:, end)' * test_data' >= theta_batch(end);

acc_online = sum(y_online == out_test)/length(out_test) * 100;
acc_batch = sum(y_batch == out_test)/length(out_test) * 100;

fprintf('Accuracy of classification train by online learning: %d  \n', acc_online);
fprintf('Accuracy of classification train by batch learning: %d \n', acc_batch);

%% 

function scatter_features(data1, data2, data3, f1, f2, f3)
    % Plotting features: sepal length vs sepal width
    scatter3(data1(:,f1), data1(:,f2), data1(:,f3), 'filled', 'MarkerFaceColor', 'r');
    hold on;
    scatter3(data2(:,f1), data2(:,f2), data2(:,f3), 'filled', 'MarkerFaceColor', 'g');
    scatter3(data3(:,f1), data3(:,f2), data3(:,f3), 'filled', 'MarkerFaceColor', 'b');
%     xlabel(sprintf('feature %d', f1));
%     ylabel(sprintf('feature %d', f2));
%     title(sprintf('feature %d vs. feature %d', f1, f2));
    legend('Iris-setosa', 'Iris-versicolor', 'Iris-virginica');
    hold off;
end

function [w_array, theta_array, iter_num] = online_learning(eta, n_iter, x, out)
    
    init_w = [0; 0];
    init_theta = 5;

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

function [w_array, theta_array, iter_num] = batch_learning(eta, n_iter, x, out)
    
    init_w = [0; 0];
    init_theta = 5;

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

function plot_line(w1, w2, theta)
    slope = -w1/w2;
    intercept = theta/w2;

    x = 0:0.1:5;
    y = slope*x + intercept;

    plot(x,y,'r'); xlabel('x'); ylabel('y');
end