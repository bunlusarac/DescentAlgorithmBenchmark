% This script will run multiple trials of optimization for each algorithm
% and log the data for further processing to benchmark the algorithms.


clc
beep on

% Dixon-Price's function
syms x1 x2
f = (x1-1)^2 + 2*(2*x2^2-x1)^2;
vars = [x1, x2];


% Exact optimum point
X1star = 1;
X2star = 1/sqrt(2);
Xstar = [X1star, X2star];

% −10 ≤ xi ≤ 10 where i=1,2
xmax = 10;
xmin = -10;

% Hessian matrix of the function
hess = hessian(f,vars);
grad = gradient(f,vars);

% Error bound
epsilon = 10^-4;
antidiverge = false;

% Maximum number of iterations until termination
% (when the algorithm hits a plateu)
max_iter = 100;

% -----------------------------------------------------------------------
% Benchmarking data

% Number of trials
n_trials = 110;

% Matrix of all starting points per trial

init_pts = zeros(n_trials, 2);

% Vectors of optimal values achieved by all algorithms from trials
nr_opt_val = zeros(n_trials, 1);
hs_opt_val = zeros(n_trials, 1);
pr_opt_val = zeros(n_trials, 1);
fr_opt_val = zeros(n_trials, 1);
qn_opt_val = zeros(n_trials, 1);

% Matrix of optimum points achieved by all algorithms from trials
nr_opt_pts = zeros(n_trials, 2);
hs_opt_pts = zeros(n_trials, 2);
pr_opt_pts = zeros(n_trials, 2);
fr_opt_pts = zeros(n_trials, 2);
qn_opt_pts = zeros(n_trials, 2);

% Vectors of execution times achieved by all algorithms from trials
nr_exc_tim = zeros(n_trials, 1);
hs_exc_tim = zeros(n_trials, 1);
pr_exc_tim = zeros(n_trials, 1);
fr_exc_tim = zeros(n_trials, 1);
qn_exc_tim = zeros(n_trials, 1);

% Iteration numbers for all trials
nr_iters = zeros(n_trials, 1);
hs_iters = zeros(n_trials, 1);
pr_iters = zeros(n_trials, 1);
fr_iters = zeros(n_trials, 1);
qn_iters = zeros(n_trials, 1);

% Do not touch this
too_many_iters = false;
%skipped_iters = zeros(n_trials,1);
skipped_iters = [];
%n_skipped = 0;

% -----------------------------------------------------------------------
% Benchmarking code

for n_trial = 1:n_trials
    disp("Performing trial: ");
    disp(n_trial);
    skip_trial = false;

    % Initialize the inital point for (x1,x2)
    % (from uniform distribution in range -10 and 10)
    
    X1 = (xmax-xmin).*rand() + xmin;
    X2 = (xmax-xmin).*rand() + xmin;
    Xinit = [X1; X2];

    init_pts(n_trial, :) = Xinit.';

    % Newton-Raphson Method
    tic
    current_pt = Xinit;
    iter = 0;
    g = vpa(subs(grad, vars, current_pt.'));
    
    while norm(g) > epsilon
        iter = iter + 1;
        
        if iter > max_iter
            disp("Reached maximum number of iterations! ")
            skip_trial = true;
            break
        end

        subhess = vpa(subs(hess,vars,current_pt.'));
        %inverse = inv(subhess);
        g = vpa(subs(grad, vars, current_pt.'));
        %current_pt = vpa(current_pt - inverse * g);
        current_pt = vpa(current_pt - subhess\g);

        if any(abs(current_pt) > xmax) && antidiverge
            disp("Diverged out of the domain! Terminating")
            return
        end
    
        %disp(g);
    end

    exectime = toc;

    if skip_trial
        disp("!!!!!!!!! Skipping trial: ")
        disp(n_trial)
        skipped_iters(end+1) = n_trial;
        continue
    end

    
    nr_iters(n_trial) = iter;
    nr_opt_val(n_trial) = vpa(subs(f, vars, current_pt.'));
    nr_opt_pts(n_trial, :) = current_pt.';
    nr_exc_tim(n_trial) = exectime;
 
    % ---------------------------------------------------------------------
    % Hestenes-Stiefel Method
    tic
    current_pt = Xinit;
    iter = 0;
    g = vpa(subs(grad, vars, current_pt.'));
    d = -g;
    
    while norm(g) > epsilon
        iter = iter + 1;
        if iter > max_iter
            disp("Reached maximum number of iterations! ")
            skip_trial = true;
            break
        end

        subhess = vpa(subs(hess,vars,current_pt.'));
        alpha = -((g.'*d)/(d.'*subhess*d));
        current_pt = vpa(current_pt + alpha.*d);
        g_old = g;
        g = vpa(subs(grad, vars, current_pt.'));
    
        %update d
        beta = (g.'*(g-g_old))/(d.'*(g-g_old));
        d = -g + beta .* d;
    end
    exectime = toc;

    if skip_trial
        disp("!!!!!!!!! Skipping trial: ")
        disp(n_trial)
        skipped_iters(end+1) = n_trial;
        continue
    end

    
    hs_iters(n_trial) = iter;
    hs_opt_val(n_trial) = vpa(subs(f, vars, current_pt.'));
    hs_opt_pts(n_trial, :) = current_pt.';
    hs_exc_tim(n_trial) = exectime;
    % ---------------------------------------------------------------------
    % Polak-Ribiere Method
    tic
    current_pt = Xinit;
    iter = 0;
    g = vpa(subs(grad, vars, current_pt.'));
    d = -g;
    
    while norm(g) > epsilon
        iter = iter + 1;
        if iter > max_iter
            disp("Reached maximum number of iterations! Terminating")
            skip_trial = true;
            break
        end

        subhess = vpa(subs(hess,vars,current_pt.'));
        alpha = -((g.'*d)/(d.'*subhess*d));
        current_pt = vpa(current_pt + alpha.*d);
        g_old = g;
        g = vpa(subs(grad, vars, current_pt.'));
    
        %update d
        beta = (g.'*(g-g_old))/(g_old.'*g_old);
        d = -g + beta .* d;
    end

    exectime = toc;

    if skip_trial
        disp("!!!!!!!!! Skipping trial: ")
        disp(n_trial)
        skipped_iters(end+1) = n_trial;
        continue
    end

    pr_iters(n_trial) = iter;
    pr_opt_val(n_trial) = vpa(subs(f, vars, current_pt.'));
    pr_opt_pts(n_trial, :) = current_pt.';
    pr_exc_tim(n_trial) = exectime;
    % -----------------------------------------------------------------------
    % Fletcher-Reeves Method
    tic
    current_pt = Xinit;
    iter = 0;
    g = vpa(subs(grad, vars, current_pt.'));
    d = -g;
    
    while norm(g) > epsilon
        iter = iter + 1;
        if iter > max_iter
            disp("Reached maximum number of iterations! ")
            skip_trial = true;
            break
        end

        subhess = vpa(subs(hess,vars,current_pt.'));
        alpha = -((g.'*d)/(d.'*subhess*d));
        current_pt = vpa(current_pt + alpha.*d);
        g_old = g;
        g = vpa(subs(grad, vars, current_pt.'));
    
        %update d
        beta = (g.'*g)/(g_old.'*g_old);
        d = -g + beta .* d;
    end
    exectime = toc;

    if skip_trial
        disp("!!!!!!!!! Skipping trial: ")
        disp(n_trial)
        skipped_iters(end+1) = n_trial;
        continue
    end

    
    fr_iters(n_trial) = iter;
    fr_opt_val(n_trial) = vpa(subs(f, vars, current_pt.'));
    fr_opt_pts(n_trial, :) = current_pt.';
    fr_exc_tim(n_trial) = exectime;
    % -----------------------------------------------------------------------
    % Quasi-Newton Method with Rank-One Correction
    tic
    current_pt = Xinit;
    iter = 0;
    g = vpa(subs(grad, vars, current_pt.'));
    
    d = -g;
    H = eye(2);
    
    while norm(g) > epsilon
        iter = iter + 1;
        if iter > max_iter
            disp("Reached maximum number of iterations! ")
            skip_trial = true;
            break
        end

        subhess = vpa(subs(hess,vars,current_pt.'));
        alpha = -(g.'*d)/(d.'*subhess*d);
        delta_x = (alpha.*d);
        current_pt = vpa(current_pt + delta_x);
        g_new = vpa(subs(grad, vars, current_pt.'));
        delta_g = g_new - g;
        H = H + (((delta_x - H * delta_g)*(delta_x - H * delta_g).')/(delta_g.' * (delta_x - H * delta_g)));
        d = -H * g_new;
        g = g_new;
        
        if any(abs(current_pt) > xmax) && antidiverge
            disp("Diverged out of the domain! Terminating")
            return
        end
   
    end
    exectime = toc;

    if skip_trial
        disp("!!!!!!!!! Skipping trial: ")
        disp(n_trial)
        skipped_iters(end+1) = n_trial;
        continue
    end


    qn_iters(n_trial) = iter;
    qn_opt_val(n_trial) = vpa(subs(f, vars, current_pt.'));
    qn_opt_pts(n_trial, :) = current_pt.';
    qn_exc_tim(n_trial) = exectime;
end

% -----------------------------------------------------------------------

opt_val_data = [nr_opt_val, hs_opt_val, pr_opt_val, fr_opt_val, qn_opt_val];
exc_tim_data = [nr_exc_tim, hs_exc_tim, pr_exc_tim, fr_exc_tim, qn_exc_tim];
iter_data = [nr_iters, hs_iters, pr_iters, fr_iters, qn_iters];

opt_x_data = [nr_opt_pts(:,1), hs_opt_pts(:,1), pr_opt_pts(:,1), fr_opt_pts(:,1), qn_opt_pts(:,1)];
opt_y_data = [nr_opt_pts(:,2), hs_opt_pts(:,2), pr_opt_pts(:,2), fr_opt_pts(:,2), qn_opt_pts(:,2)];

for ski_idx = 1:length(skipped_iters)
    idx = skipped_iters(ski_idx);
    exc_tim_data(idx,:) = [];
    opt_val_data(idx,:) = [];
    opt_x_data(idx,:) = [];
    opt_y_data(idx,:) = [];
    iter_data(idx,:) = [];

    for upd_ski_idx = (ski_idx + 1):length(skipped_iters)
        skipped_iters(upd_ski_idx) = skipped_iters(upd_ski_idx) - 1;
    end
end

nr_exc_tim_mean = mean(nr_exc_tim);
hs_exc_tim_mean = mean(hs_exc_tim);
pr_exc_tim_mean = mean(pr_exc_tim);
fr_exc_tim_mean = mean(fr_exc_tim);
qn_exc_tim_mean = mean(qn_exc_tim);

nr_opt_val_mean = mean(nr_opt_val);
hs_opt_val_mean = mean(hs_opt_val);
pr_opt_val_mean = mean(pr_opt_val);
fr_opt_val_mean = mean(fr_opt_val);
qn_opt_val_mean = mean(qn_opt_val);


%opt_val_tbl = array2table(opt_val_data', 'VariableNames' ,["Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"])
%exc_tim_tbl = array2table(exc_tim_data', 'VariableNames' ,["Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"])

%opt_x_tbl = array2table(opt_x_data, 'VariableNames',["Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"])
%opt_y_tbl = array2table(opt_y_data, 'VariableNames',["Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"])

figure(1)
%alg = categorical(exc_tim_tbl,1:5,["Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"])
boxchart(opt_val_data)
hold on
title("Function Value at Found Optimum")
ylabel("f(x^*)")
set(gca,'XTickLabel',{"Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"});
hold off

figure(2)
%alg = categorical(exc_tim_tbl,1:5,["Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"])
boxchart(exc_tim_data)
hold on
title("Execution Time Until Optimized")
ylabel("Execution time (seconds)")
set(gca,'XTickLabel',{"Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"});

figure(3)
boxchart(opt_x_data)
hold on
title("Found x^*_1 values")
ylabel("x^*_1")
set(gca,'XTickLabel',{"Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"});
hold off

figure(4)
boxchart(opt_y_data)
hold on
title("Found x^*_2 values")
ylabel("x^*_2")
set(gca,'XTickLabel',{"Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"});
hold off

figure(5)
boxchart(iter_data)
hold on
title("Number of Iterations Until Optimized")
ylabel("n_{iterations}")
set(gca,'XTickLabel',{"Newton-Raphson", "Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves", "Quasi-Newton"});
hold off

disp("Total number of successful trials: " )
disp(n_trials - length(skipped_iters))

 for alg = 1:5
x1min = min(opt_x_data(:,alg));
x1max = max(opt_x_data(:,alg));
x1mean = mean(opt_x_data(:,alg));
x2min = min(opt_y_data(:,alg));
x2max = max(opt_y_data(:,alg));
x2mean = mean(opt_y_data(:,alg));
xstarmin = min(opt_val_data(:,alg));
xstarmax = max(opt_val_data(:,alg));
xstarmean = mean(opt_val_data(:,alg));
itermin = min(iter_data(:,alg));
itermax = max(iter_data(:,alg));
itermean = mean(iter_data(:,alg));
exctmin = min(exc_tim_data(:,alg));
exctmax = max(exc_tim_data(:,alg));
exctmean = mean(exc_tim_data(:,alg));
fprintf("Algo %d -- \n %.4f %.4f %.4f \n%.4f %.4f %.4f\n%.4f %.4f %.4f\n%.4f %.4f %.4f\n%.4f %.4f %.4f\n", alg, x1mean, x1min, x1max, x2mean, x2min, x2max, xstarmean, xstarmin, xstarmax, itermean, itermin, itermax, exctmean, exctmin, exctmax);
end