% Define the number of clusters (K) and the data
K = 5;  % You can change this based on your problem

load('DataNew.mat')
data = DataNew';

% Define the bounds for the centroids (based on the data range)
lb = min(data);
ub = max(data);

% Define the genetic algorithm options
options = optimoptions('ga', 'Display', 'iter', 'MaxGenerations', 100, 'PopulationSize', 50);

% Define the fitness function for kmeans
fitnessFunction = @(centroids) kmeansFitness(data, centroids, K);

% Run the genetic algorithm
centroidsGA = ga(fitnessFunction, size(data, 2) * K, [], [], [], [], lb, ub, [], options);

% Reshape the centroids for kmeans
centroidsGA = reshape(centroidsGA, K, []);

% Compute distances from each data point to each centroid
distancesGA = pdist2(data, centroidsGA);

% Find the closest centroid for each data point
[~, idxGA] = min(distancesGA, [], 2);

% Visualize the results
figure;
scatter(data(:, 1), data(:, 2), 30, idxGA, 'filled');
hold on;
scatter(centroidsGA(:, 1), centroidsGA(:, 2), 100, 'k', 'x', 'LineWidth', 2);
title('Clusters with Optimized Centroids');
xlabel('Feature 1');
ylabel('Feature 2');
colormap(gca, jet(K));

%% Functions

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
