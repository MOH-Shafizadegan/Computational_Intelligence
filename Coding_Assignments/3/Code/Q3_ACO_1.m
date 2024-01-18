clc; clear; close all;

% Parameters
numClusters = 5;
numAnts = 10;
maxIterations = 100;

load('DataNew.mat')
data = DataNew';

% Apply ACO for clustering
[bestSolution, bestCentroids] = antColonyClustering(data, numClusters, numAnts, maxIterations);

% Visualize the results
figure;

gscatter(data(:, 1), data(:, 2), bestSolution, 'rgbcmyk');
hold on;
plot(bestCentroids(:, 1), bestCentroids(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', 3);
title('ACO Clustering');
hold off;


%%

function [bestSolution, bestCentroids] = antColonyClustering(data, numClusters, numAnts, maxIterations)
    % Initialize pheromones (not used in this example)
    pheromones = ones(size(data, 1), numClusters) / numClusters;

    % Main loop
    for iteration = 1:maxIterations
        solutions = zeros(numAnts, size(data, 1));

        % Ant solutions construction
        for ant = 1:numAnts
            solutions(ant, :) = constructSolution(pheromones);
        end

        % Pheromone update
        pheromones = updatePheromones(pheromones, solutions, data);
        
        % Find the best solution
        [minDistance, minIndex] = min(evaluateSolutions(solutions, data));
        bestSolution = solutions(minIndex, :);
        bestCentroids = calculateCentroids(data, bestSolution, numClusters);
    end
end

function solution = constructSolution(pheromones)
    % Construct a solution using pheromones for k-means clustering

    numDataPoints = size(pheromones, 1);

    % Initialize probabilities for each data point to be assigned to each cluster
    probabilities = pheromones;

    % Normalize probabilities
    probabilities = probabilities ./ sum(probabilities, 2);

    % Assign each data point to a cluster based on probabilities
    solution = zeros(1, numDataPoints);
    for point = 1:numDataPoints
        solution(point) = selectCluster(probabilities(point, :));
    end
end

function selectedCluster = selectCluster(probabilities)
    % Select a cluster based on the probabilities
    cumProbabilities = cumsum(probabilities);
    randomValue = rand();
    selectedCluster = find(cumProbabilities >= randomValue, 1);
end


function centroids = calculateCentroids(data, solution, numClusters)
    % Calculate cluster centroids based on the solution
    centroids = zeros(numClusters, size(data, 2));
    for cluster = 1:numClusters
        clusterPoints = data(solution == cluster, :);
        centroids(cluster, :) = mean(clusterPoints, 1);
    end
end

function distances = evaluateSolutions(solutions, data)
    % Evaluate the solutions based on the sum of squared distances
    distances = zeros(size(solutions, 1), 1);
    for ant = 1:size(solutions, 1)
        centroids = calculateCentroids(data, solutions(ant, :), max(solutions(ant, :)));
        distances(ant) = sum((data - centroids(solutions(ant, :), :)).^2, 'all');
    end
end

function newPheromones = updatePheromones(pheromones, solutions, data)
    % Update pheromones based on the assigned clusters
    evaporationRate = 0.05;
    depositRate = 0.8;

    % Evaporation
    pheromones = (1 - evaporationRate) * pheromones;

    for ant = 1:size(solutions, 1)
        centroids = calculateCentroids(data, solutions(ant, :), max(solutions(ant, :)));
        
        for point = 1:size(data, 1)
            cluster = solutions(ant, point);
            temp = min(norm(data(point, :) - centroids(cluster, :))^2);
            pheromones(point, cluster) = pheromones(point, cluster) + 10 * depositRate / temp;
        end
    end
    
    % Normalize pheromones
    newPheromones = pheromones ./ sum(pheromones, 2);
end