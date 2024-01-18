clc; clear; close all;

load('SampleData.mat')

%%
clc;

% Separate the data into two classes based on the labels
class1Data = TrainingData(:, TrainingLabels == 0);
class2Data = TrainingData(:, TrainingLabels == 1);

% Plot the data of two classes
figure;
scatter(class1Data(1, :), class1Data(2, :), 'ro', 'filled'); 
hold on; % Hold on to the current figure
scatter(class2Data(1, :), class2Data(2, :), 'bo', 'filled');

% Add labels and title
xlabel('Feature 1');
ylabel('Feature 2');
title('Data of Two Classes');

% Add a legend
legend('Class 1', 'Class 2');

hold off;

%%
clc;

% Split into training and validation sets (70% training, 30% validation)
[trainInd, valInd, ~] = dividerand(size(TrainingData, 2), 0.7, 0.3, 0);
trainX = TrainingData(:, trainInd);
trainY = TrainingLabels(:, trainInd);
valX = TrainingData(:, valInd);
valY = TrainingLabels(:, valInd);

%% RBF with one output neuron
clc;

% Define the network spec
hiddenNeurons = 19;
spread = 2;

% Design the RBF network
net_1 = newrb(trainX, trainY, 0, spread, hiddenNeurons);

val_pred_1 = net_1(valX);

% Calculate the classification accuracy
predictedLabels = round(val_pred_1);
accuracy = sum(predictedLabels == valY) / numel(valY);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

%% calibrate parametes

clc;

nhidden = 1:20;
sigma = 0.1:0.1:10;
[X, Y] = meshgrid(nhidden, sigma);

acc = zeros(length(nhidden), length(sigma));
for i=1:length(nhidden)
    for j=1:length(sigma)
        acc(i,j) = pred_RBF_1out(nhidden(i), sigma(j), valX, valY, trainX, trainY);
    end
end

%%
clc;

[~, idx] = max(acc(:));
[row, col] = ind2sub(size(acc), idx);

fprintf('Best accuracy of %f achieved with nhidden = %d and sigma = %f \n', ...
         acc(row, col), nhidden(row), sigma(col));


%% Visualization
clc; close all;

figure;
surf(X, Y, acc');
xlabel('nhidden');
ylabel('sigma');
zlabel('accuracy');

%% RBF with 2 output neuron

clc;

% Define the desired number of neurons in the hidden layer
hiddenNeurons = 20;

train_Y_2 = [1-trainY; trainY];
% Design the RBF network
net_2 = newrb(trainX, train_Y_2, 0, 1, hiddenNeurons);

val_Y_2 = [1-valY; valY];
val_pred_2 = sim(net_2, valX);

% Calculate the classification accuracy
[~, predictedLabels] = max(val_pred_2);
[~, actualLabels] = max(val_Y_2);
accuracy = sum(predictedLabels == actualLabels) / numel(actualLabels);
fprintf('Validation accuracy: %.2f%%\n', accuracy * 100);

%% Optimal paremetrs
clc;

nhidden2 = 1:30;
sigma2 = 0.1:0.2:10;
[X, Y] = meshgrid(nhidden2, sigma2);

acc2 = zeros(length(nhidden2), length(sigma2));
for i=1:length(nhidden2)
    for j=1:length(sigma2)
        acc2(i,j) = pred_RBF_2out(nhidden2(i), sigma2(j), valX, valY, trainX, trainY);
    end
end

%%

clc;

[~, idx] = max(acc2(:));
[row2, col2] = ind2sub(size(acc2), idx);

fprintf('Best accuracy of %f achieved with nhidden = %d and sigma = %f \n', ...
         acc2(row2, col2), nhidden2(row2), sigma2(col2));

%% Visualization
clc; close all;

figure;
surf(X, Y, acc2');
xlabel('nhidden');
ylabel('sigma');
zlabel('accuracy');

%% Functions

function acc = pred_RBF_1out(nhidden, sigma, valX, valY, trainX, trainY)

    % Design the RBF network
    net_1 = newrb(trainX, trainY, 0, sigma, nhidden);

    val_pred_1 = net_1(valX);

    % Calculate the classification accuracy
    predictedLabels = round(val_pred_1);
    acc = sum(predictedLabels == valY) / numel(valY);

end

function acc = pred_RBF_2out(nhidden, sigma, valX, valY, trainX, trainY)

    train_Y_2 = [1-trainY; trainY];
    % Design the RBF network
    net_2 = newrb(trainX, train_Y_2, 0, sigma, nhidden);

    val_Y_2 = [1-valY; valY];
    val_pred_2 = sim(net_2, valX);

    % Calculate the classification accuracy
    [~, predictedLabels] = max(val_pred_2);
    [~, actualLabels] = max(val_Y_2);
    acc = sum(predictedLabels == actualLabels) / numel(actualLabels);

end



