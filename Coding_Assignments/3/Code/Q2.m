clc; clear; close all;

load('DataNew.mat')

k = 5;
indices = randperm(size(DataNew, 2), k);
init_cent = DataNew(:, indices);
[centroids, labels] = my_kmeans(DataNew, k, init_cent);

%% Visualization

plot_clusters(k, DataNew, centroids, labels)

%% 4 clusters

k = 4;
indices = randperm(size(DataNew, 2), k);
init_cent = DataNew(:, indices);
[centroids, labels] = my_kmeans(DataNew, k, init_cent);

plot_clusters(k, DataNew, centroids, labels)

%% 6 clusters

k = 6;
indices = randperm(size(DataNew, 2), k);
init_cent = DataNew(:, indices);
[centroids, labels] = my_kmeans(DataNew, k, init_cent);

plot_clusters(k, DataNew, centroids, labels)

%% MATLAB kmeans
clc;

k=5;
[labels, centroids] = kmeans(DataNew', k);
plot_clusters(k, DataNew, centroids', labels)

%%
clc;

k=4;
[labels, centroids] = kmeans(DataNew', k);
plot_clusters(k, DataNew, centroids', labels)

%%
clc;

k=6;
[labels, centroids] = kmeans(DataNew', k);
plot_clusters(k, DataNew, centroids', labels)

%% Hirerachial clustering
clc;

data = DataNew'; 

%Perform hierarchical clustering
linkageMatrix = linkage(data, 'ward', 'euclidean'); 

% Plot the dendrogram
dendrogram(linkageMatrix);

k = 5; % Specify the desired number of clusters
clusterLabels = cluster(linkageMatrix, 'maxclust', k);
plot_clusters_2(k, data', clusterLabels);

%%
k = 4; % Specify the desired number of clusters
clusterLabels = cluster(linkageMatrix, 'maxclust', k);
plot_clusters_2(k, data', clusterLabels);

%%

k = 6; % Specify the desired number of clusters
clusterLabels = cluster(linkageMatrix, 'maxclust', k);
plot_clusters_2(k, data', clusterLabels);

%% Functions

function [centroids, labels] = my_kmeans(data, k, init_cent)

    centroids = init_cent;

    % Initialize labels and previous centroids
    labels = zeros(size(data, 2), 1);
    prevCentroids = zeros(size(centroids));

    % Iterate until convergence
    while ~isequal(centroids, prevCentroids)
        % Assign each data point to the nearest centroid
        for i = 1:size(data, 2)
            distances = sum((data(:, i) - centroids).^2, 1);  % Adjusted indexing
            [~, index] = min(distances);
            labels(i) = index;
        end

        % Update the centroids
        prevCentroids = centroids;
        for j = 1:k
            clusterPoints = data(:, labels == j);
            centroids(:, j) = mean(clusterPoints, 2);  % Adjusted mean calculation
        end
    end
end

function plot_clusters(k, data, centroids, labels)

    figure;
    hold on;
    colors = lines(k);
    for i = 1:k
        clusterPoints = data(:, labels == i);
        scatter(clusterPoints(1, :), clusterPoints(2, :), [], colors(i, :), 'filled');
    end
    scatter(centroids(1, :), centroids(2, :), 100, 'kx', 'linewidth', 3);
    hold off;
    title('k-means Clustering');

end

function plot_clusters_2(k, data, labels)

    figure;
    hold on;
    colors = lines(k);
    for i = 1:k
        clusterPoints = data(:, labels == i);
        scatter(clusterPoints(1, :), clusterPoints(2, :), [], colors(i, :), 'filled');
    end
    hold off;
    title('Hierarchial Clustering');

end