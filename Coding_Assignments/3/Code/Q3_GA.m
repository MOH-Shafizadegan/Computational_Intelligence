clear; clc; close all;

% Define the maximum number of clusters (K)
maxClusters = 5;

load('DataNew.mat')
data = DataNew';

% Define the bounds for the centroids (based on the data range)
lb = min(data);
ub = max(data);

% Define the genetic algorithm options
options = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 100, 'PopulationSize', 50);

% Define the fitness function for kmeans
fitnessFunction = @(params) kmeansFitness(data, params, maxClusters);

% Run the genetic algorithm
numGenes = size(data, 2) * maxClusters;
centroidsGA = ga(fitnessFunction, numGenes, [], [], [], [], lb, ub, [], options);

% Determine the number of clusters based on the data
K = max(1, min(size(centroidsGA, 1) / size(data, 2), maxClusters));

% Reshape the centroids for kmeans
centroidsGA = reshape(centroidsGA, K, []);

% Compute distances from each data point to each centroid
distancesGA = pdist2(data, centroidsGA);

% Find the closest centroid for each data point
[~, idxGA] = min(distancesGA, [], 2);

% Visualize the results
figure;

% Plot the data points with original clusters
subplot(1, 2, 1);
scatter(data(:, 1), data(:, 2), 30, idx, 'filled');
title('Original Clusters');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(gca, jet(max(idx)));

% Plot the data points with optimized clusters
subplot(1, 2, 2);
scatter(data(:, 1), data(:, 2), 30, idxGA, 'filled');
hold on;
scatter(centroidsGA(:, 1), centroidsGA(:, 2), 100, 'k', 'x', 'LineWidth', 2);
title('Clusters with Optimized Centroids');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(gca, jet(K));

% Define the fitness function for kmeans
function fitness = kmeansFitness(data, params, maxClusters)
    % Reshape the centroids based on the number of clusters
    K = max(1, min(size(params, 1) / size(data, 2), maxClusters));
    centroids = reshape(params, K, size(data, 2));

    % Compute distances from each data point to each centroid
    distances = pdist2(data, centroids);

    % Find the closest centroid for each data point
    [~, idx] = min(distances, [], 2);

    % Compute the sum of squared distances
    fitness = sum(sum((data - centroids(idx, :)).^2));
end
