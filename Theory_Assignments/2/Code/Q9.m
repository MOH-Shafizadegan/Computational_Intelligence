clc; clear; close all;

dx = 0.01;
x = -0.2:dx:4.2;

f = @(x) 5/6 * x.^4 - 7*x.^3 + 115/6 * x.^2 - 18.*x + 6;

figure;
plot(x, f(x), 'linewidth', 2);
title('Cost function')
xlabel('x');
ylabel('f(x)')
 

%% Part b

clc; close all;

syms x;
fx = 5/6 * x.^4 - 7*x.^3 + 115/6 * x.^2 - 18.*x + 6;
df = matlabFunction(diff(f, x));

avg = 2.05;
init_x = randn() + avg;

learningRate = 0.02;
n_iter = 100;

% Perform gradient descent
x = init_x;
for i = 1:n_iter
    
    df_val = df(x);
    % Update the current point using the derivative and learning rate
    x = x - learningRate * df_val;
    
    % Check for convergence (i.e., when the derivative is close to zero)
    if abs(df_val) < 1e-2
        break;
    end
end

% Display the final result
disp(['The initial value: x = ', num2str(init_x)])
disp(['Number of iterations: ', num2str(i)])
disp(['Minimum point: x = ', num2str(x)]);
disp(['Minimum value: f(x) = ', num2str(f(x))]);

%% part b (continued)

clc;

eta = [0.01, 0.02, 0.04, 0.08, 0.1];
avg = 2.05;
max_iter = 100;
    
for i=1:length(eta)
   
    init_x = randn() + avg;

    [n_iter, min_x, min_v] = gradient_descent(init_x, eta(i), max_iter, f);
    
    % Display the final result
    disp(['Learning rate = ', num2str(eta(i))])
    disp(['The initial value: x = ', num2str(init_x)])
    disp(['Number of iterations: ', num2str(n_iter)])
    disp(['Minimum point: x = ', num2str(min_x)]);
    disp(['Minimum value: f(x) = ', num2str(min_v)]);
    disp('-----------------------------')
   
end

%% part c

clc; close all;

global_min = min(f(x));
avg = 2.05;
prob = zeros(1, length(eta));
max_iter = 200;

for i=1:length(eta)
   
    N = 100;
    n_true_min = 0;
    
    for j=1:N
        init_x = randn() + avg;
        [n_iter, min_x, min_v] = gradient_descent(init_x, eta(i), max_iter, f);
        if abs(min_v - global_min) < 1e-5
           n_true_min = n_true_min + 1;
        end
    end
        
    prob(i) = n_true_min / N;
    
end

figure;
bar(eta, prob)
xlabel('eta')
ylabel('Probability')
title('Probability of reaching the global min')
xlim([-0.02 0.12])
ylim([0 1])

%% part d

clc; close all;

global_min = min(f(x));
avg = 0:0.5:4;
prob = zeros(1, length(avg));
eta = 0.02;
max_iter = 200;

for i=1:length(avg)
   
    N = 100;
    n_true_min = 0;
    
    for j=1:N
        init_x = randn() + avg(i);
        [n_iter, min_x, min_v] = gradient_descent(init_x, eta, max_iter, f);
        if abs(min_v - global_min) < 1e-5
           n_true_min = n_true_min + 1;
        end
    end
        
    prob(i) = n_true_min / N;
    
end

figure;
bar(avg, prob)
xlabel('average')
ylabel('Probability')
title('Probability of reaching the global min')
xlim([-0.5 4.5])
ylim([0 1])

%% part e

clc; close all;

eta = [0.01, 0.02, 0.04, 0.08, 0.1, 0.15, 0.2, 0.4];
global_min = min(f(x));
avg = 2.05;
prob = zeros(1, length(eta));
max_iter = 50;

for i=1:length(eta)
   
    N = 100;
    n_diverge = 0;
    
    for j=1:N
        init_x = randn() + avg;
        [n_iter, min_x, min_v, flag] = modified_gradient_descent(init_x, eta(i), max_iter, f);
        if ~flag
           n_diverge = n_diverge + 1;
        end
    end
        
    prob(i) = n_diverge / N;
    
end

figure;
bar(eta, prob)
xlabel('eta')
ylabel('Probability')
title('Probability of divergence')
xlim([-0.02 0.45])
ylim([0 1])


%% Functions

function [i, min_x, min_v] = gradient_descent(init_x, eta, n_iter, f)

    syms x
    df = matlabFunction(diff(f, x));
    
    % Perform gradient descent
    x = init_x;
    for i = 1:n_iter

        df_val = df(x);
        % Update the current point using the derivative and learning rate
        x = x - eta * df_val;

        % Check for convergence (i.e., when the derivative is close to zero)
        if abs(df_val) < 1e-2
            break;
        end
    end
    
    min_x = x;
    min_v = f(x);
    
end

function [i, min_x, min_v, flag] = modified_gradient_descent(init_x, eta, n_iter, f)

    syms x
    df = matlabFunction(diff(f, x));
    
    flag = 0; % not converged
    % Perform gradient descent
    x = init_x;
    for i = 1:n_iter

        df_val = df(x);
        % Update the current point using the derivative and learning rate
        x = x - eta * df_val;

        % Check for convergence (i.e., when the derivative is close to zero)
        if abs(df_val) < 1e-2
            flag = 1; % converged
            break;
        end
    end
    
    min_x = x;
    min_v = f(x);
    
end