clc; clear; close all;

% Set up the k-means parameters
k = 5; % Number of clusters

load('DataNew.mat')
data = DataNew';

% Set up PSO parameters
options = optimoptions('particleswarm', 'SwarmSize', 50, 'MaxIterations', 100);

% Define the objective function handle with extra parameters (X, k)
objectiveFunction = @(positions) kmeansFitness(data, positions, k);

% Run PSO to find cluster centroids
numVariables = k * size(data, 2);
lb = repelem(min(data), k);
ub = repelem(max(data), k);

[bestPositions, cost] = particleswarm(objectiveFunction, numVariables, lb, ub, options);

% Reshape the best positions to get the final cluster centroids
numElementsPerCluster = numVariables / k;
bestCentroids = reshape(bestPositions(1 : floor(numElementsPerCluster) * k), k, []);


% Assign data points to the final centroids
[~, clusterIndices] = pdist2(bestCentroids, data, 'euclidean', 'Smallest', 1);

% Plot the results
figure;
scatter(data(:, 1), data(:, 2), 50, clusterIndices, 'filled');
hold on;
scatter(bestCentroids(:, 1), bestCentroids(:, 2), 200, 'rx', 'LineWidth', 2);
title('K-Means Clustering with PSO');

%% Functions

% Define the fitness function for kmeans
function fitness = kmeansFitness(data, centroids, K)
    % Reshape the centroids
    centroids = reshape(centroids, K, []);

    % Compute distances from each data point to each centroid
    distances = pdist2(data, centroids);

    % Find the closest centroid for each data point
    [~, idx] = min(distances, [], 2);

    % Compute the sum of squared distances
    fitness = sum(sum((data - centroids(idx, :)).^2));
end


