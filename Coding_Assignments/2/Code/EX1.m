clc; clear; close all;

% load data
data = load('../Data/Ex1.mat');
X = [data.NOemission' data.speed'];
Y = data.fuelrate';

%% Part a: scatter plot

figure;
scatter3(X(:, 1), X(:, 2), Y, 'filled');
xlabel('NO Emission');
ylabel('Speed');
zlabel('Fuel Rate');
title('Scatter Plot of NO Emission, Speed, and Fuel Rate');

%% Part b: training and validation sets

trainX = X(1:700, :);
trainY = Y(1:700);
validationX = X(701:end, :);
validationY = Y(701:end);

%% Part c: Linear Regression

clc; close all;

% Fitting a linear regression model using the training set
linear_model = fitlm(trainX, trainY);

% Predicting the fuel rate for the training and validation sets
trainY_pred = predict(linear_model, trainX);
validationY_pred = predict(linear_model, validationX);

% Calculating MSE for the training and validation sets
trainMSE = mean((trainY - trainY_pred).^2);
validationMSE = mean((validationY - validationY_pred).^2);

disp(['Training Set MSE: ' num2str(trainMSE)]);
disp(['Validation Set MSE: ' num2str(validationMSE)]);

%% Visualize the resulting linear regression plane

% Extracting the coefficients and intercept from the linear regression model
coefficients = linear_model.Coefficients.Estimate(2:end);
intercept = linear_model.Coefficients.Estimate(1);

% Creating a meshgrid for the hyperplane visualization
x1 = linspace(min(X(:, 1)), max(X(:, 1)), 100);
x2 = linspace(min(X(:, 2)), max(X(:, 2)), 100);
[X1, X2] = meshgrid(x1, x2);
Y_pred = coefficients(1) * X1 + coefficients(2) * X2 + intercept;

% Scatter plot of the data points
figure;
scatter3(X(:, 1), X(:, 2), Y, 'filled');
hold on;

% Plotting the linear regression hyperplane
surf(X1, X2, Y_pred, 'FaceAlpha', 0.5, 'EdgeColor', 'none');

xlabel('NO Emission');
ylabel('Speed');
zlabel('Fuel Rate');
title('Linear Regression Hyperplane');

hold off;

%% Part d: Logistic Regression

clc; close all;

Y = max(1.1*trainY);
log_trainY = log((Y - trainY)./trainY);

% Fitting a logistic regression model using the training set
logistic_model = fitlm(trainX, log_trainY);

% Predicting the fuel rate probabilities for the training and validation sets
trainY_pred = predict(logistic_model, trainX);
trainY_pred = Y ./ (1 + exp(trainY_pred));
validationY_pred = predict(logistic_model, validationX);
validationY_pred = Y ./ (1 + exp(validationY_pred));

% Calculating MSE for the training and validation sets
trainMSE = mean((trainY - trainY_pred).^2);
validationMSE = mean((validationY - validationY_pred).^2);

disp(['Training Set MSE: ' num2str(trainMSE)]);
disp(['Validation Set MSE: ' num2str(validationMSE)]);

%% Visualize logestic regression model

% Extracting model parameters
coef = logistic_model.Coefficients.Estimate;
intercept = coef(1);
weights = coef(2:end);

% Plotting the data points
figure;
scatter3(trainX(:, 1), trainX(:, 2), trainY, 'filled');
hold on;

% Creating a grid of points to generate the decision boundary
x1 = linspace(min(trainX(:, 1)), max(trainX(:, 1)), 100);
x2 = linspace(min(trainX(:, 2)), max(trainX(:, 2)), 100);
[X1, X2] = meshgrid(x1, x2);
X_grid = [X1(:), X2(:)];

% Calculating the log-odds (linear combination) for the grid points
log_odds = intercept + X_grid * weights;

% Applying the logistic (sigmoid) function to obtain probabilities
Y_grid = max(trainY) ./ (1 + exp(log_odds));

% Reshaping the predicted probabilities to match the grid dimensions
Y_grid = reshape(Y_grid, size(X1));

% Plotting the decision boundary
surf(X1, X2, Y_grid, 'EdgeColor', 'none');

% Setting plot labels and title
xlabel('NO Emission');
ylabel('Speed');
title('Logistic Regression hyperplane');

% Displaying the plot
hold off;

%% 

clc; close all;

% Define the number of neurons in the hidden layer
hiddenSize = 40;

% Create the MLP model
net = fitnet(hiddenSize, 'trainlm');

% Set the training parameters
net.trainParam.showWindow = false;  % Disable training window display
net.divideParam.trainRatio = 100/100;
net.divideParam.testRatio = 0/100;
net.divideParam.valRatio = 0/100;

% Train the MLP model
net = train(net, trainX', trainY');

% Make predictions on the training and validation sets
trainPred = net(trainX');
valPred = net(validationX');

% Calculate the MSE for the training and validation sets
trainMSE = mean((trainPred - trainY').^2);
valMSE = mean((valPred - validationY').^2);

% Display the MSE values
fprintf('Training MSE: %.4f\n', trainMSE);
fprintf('Validation MSE: %.4f\n', valMSE);

%%

clc; close all;

n_hidden = 1:20;
val_MSE_errors = zeros(1,length(n_hidden));
train_MSE_errors = zeros(1,length(n_hidden));

for i=1:length(n_hidden)
    [train_MSE_errors(i), val_MSE_errors(i)] = MLP(n_hidden(i), trainX, validationX, trainY, validationY);
end

figure;
plot(n_hidden, val_MSE_errors); hold on;
plot(n_hidden, train_MSE_errors);
legend('Validation MSE', 'Training MSE');
xlabel('number of hidden neuons')
ylabel('MSE')

%% Functions

function [trainMSE, valMSE] = MLP(n_hidden, trainX, validationX, trainY, validationY)
    
% Create the MLP model
    net = fitnet(n_hidden, 'trainlm');

    % Set the training parameters
    net.trainParam.showWindow = false;  % Disable training window display
    net.divideParam.trainRatio = 100/100;
    net.divideParam.testRatio = 0/100;
    net.divideParam.valRatio = 0/100;

    % Train the MLP model
    net = train(net, trainX', trainY');

    % Make predictions on the training and validation sets
    trainPred = net(trainX');
    valPred = net(validationX');

    % Calculate the MSE for the training and validation sets
    trainMSE = mean((trainPred - trainY').^2);
    valMSE = mean((valPred - validationY').^2);

end

