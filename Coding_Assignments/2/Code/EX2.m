clc; clear; close all;

% load data
data = load('../Data/Ex2.mat');
trainVal_data = data.TrainData;
test_data = data.TestData;

%%

% Load or define your training data and labels
% Split into training and validation sets (80% training, 20% validation)
[trainInd, valInd, ~] = dividerand(size(trainVal_data, 2), 0.8, 0.2, 0);
trainX = trainVal_data(1:3, trainInd);
trainY = trainVal_data(4, trainInd);
valX = trainVal_data(1:3, valInd);
valY = trainVal_data(4, valInd);

%% Visualization

clc; close all;

cls1_idx = find(trainY == 1);
class1X = trainX(:, cls1_idx);

cls2_idx = find(trainY == 0);
class2X = trainX(:, cls2_idx);

figure;
scatter3(class1X(1,:), class1X(2,:), class1X(3,:), 'filled'); hold on;
scatter3(class2X(1,:), class2X(2,:), class2X(3,:), 'filled');
xlabel('x1')
ylabel('x2')
zlabel('x3')


%%

% Define the number of neurons in the hidden layer
hiddenSize = 10;

% Create the MLP model
net = patternnet(hiddenSize);

net.divideParam.trainRatio = 100/100;
net.divideParam.testRatio = 0/100;
net.divideParam.valRatio = 0/100;
    
% Train the MLP model
net = train(net, trainX, trainY);

val_predict = round(net(valX));
error = sum(abs(val_predict - valY));
disp(error)

%%
clc; close all;

% Load or define your test data
testX = test_data(1:3,:);  % Test data features

% Convert the continuous output to binary labels
predictedLabels = round(net(testX));

save("Testlabel_a.mat", 'predictedLabels');

%%

clc; close all;

% Define the number of neurons in the hidden layer
hiddenSize = 10;

% Create the MLP model
net = patternnet(hiddenSize);

% Set the number of output neurons to 2
net.layers{end}.size = 2;
net.divideParam.trainRatio = 100/100;
net.divideParam.testRatio = 0/100;
net.divideParam.valRatio = 0/100;

% Train the MLP model
net = train(net, trainX, [1-trainY; trainY]);

val_predict = net(valX);
% Apply softmax function to obtain class probabilities
probabilities = softmax(val_predict);

% Determine the predicted labels based on the maximum probability
[~, valLabels] = max(probabilities);
valLabels = valLabels - 1;

error = sum(abs(valLabels - valY));
disp(error)

%% 

% Make predictions on the test data
testY = net(testX);

Y_probabilities = softmax(testY);
% Determine the predicted labels based on the maximum probability
[~, yLabels] = max(Y_probabilities);
yLabels = yLabels - 1;

save("Testlabel_b.mat", 'yLabels');
