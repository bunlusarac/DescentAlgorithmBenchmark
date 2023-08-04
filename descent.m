% This script will run a single pass of optimization for each algorithm
% and log the results.
% I will be using Dixon-Price's Function as the test function.

clc

% Abort execution if algorithms diverge out of domain
antidiverge = false;

if antidiverge
    disp("The code is adjusted to terminate on any values outside the domain.")
    disp("To turn it off, change antidiverge boolean to false")
end

% Dixon-Price's function
syms x1 x2
f = (x1-1)^2 + 2*(2*x2^2-x1)^2;
f_func = @(x1_, x2_) (x1_-1)^2 + 2*(2*x2_^2-x1_)^2; %Function version for arrayfun

vars = [x1, x2];

% −10 ≤ xi ≤ 10 where i=1,2

xmax = 10;
xmin = -10;

% Global minima
X1star = 1;
X2star_pos = 1/sqrt(2);
X2star_neg = -1/sqrt(2);
Xstar_pos = [X1star, X2star_pos];
Xstar_neg = [X1star, X2star_neg];

disp("The exact global minima are: ");
disp(Xstar_pos);
disp(Xstar_neg);

% Initialize the inital point for (x1,x2)
% (from uniform distribution in range -10 and 10)

X1 = (xmax-xmin).*rand() + xmin;
X2 = (xmax-xmin).*rand() + xmin;
Xinit = [X1; X2];

disp("The initial point is: ");
disp(Xinit);

% Hessian matrix of the function
hess = hessian(f,vars);
grad = gradient(f,vars);

% Error bound
epsilon = 10^-10;

% Maximum number of iterations, to allocate memory for points
max_iter = 50;

nr_points = zeros(max_iter, 2);
hs_points = zeros(max_iter, 2);
pr_points = zeros(max_iter, 2);
fr_points = zeros(max_iter, 2);
qn_points = zeros(max_iter, 2);

% Plots
figure(1)
hold off
%fcontour(f, [-10 10], 'LevelList',[0:5:100])
%fcontour(f, [-10 10])
fcontour(f)
hold on


% -----------------------------------------------------------------------
% Newton-Raphson Method
tic
current_pt = Xinit;
iter = 0;
g = vpa(subs(grad, vars, current_pt.'));

while norm(g) > epsilon
    iter = iter + 1;

    if(iter > max_iter)
        disp("Iterated more than max iteration number. Try to run script again. Terminating...")
        return
    end

    nr_points(iter, :) = current_pt;

    subhess = vpa(subs(hess,vars,current_pt.'));
    %inverse = inv(subhess);
    g = vpa(subs(grad, vars, current_pt.'));
    %current_pt = vpa(current_pt - inverse * g);
    current_pt = vpa(current_pt - subhess\g);
    
    if any(abs(current_pt) > xmax) && antidiverge
        disp("Diverged out of the domain! Try to run script again. Terminating...")
        disp("Diverged point:");
        disp(current_pt);
        return
    end
    
    %disp(g);
end
exectime = toc;

nr_points(iter+1, :) = current_pt;

%vpa(subs(f,vars,current_pt.'))
disp("------[Newton-Raphson Method]------");
disp("Iterations: " + iter);
disp("Optimum point: ");
disp(current_pt);
disp("Optimum value: ");
disp(vpa(subs(f, vars, current_pt.')));
disp("Gradient: ");
disp(vpa(subs(grad, vars, current_pt.')));
disp("Execution time: ");
disp(exectime);
fprintf("\n\n");

% -----------------------------------------------------------------------
% Hestenes-Stiefel Method
tic
current_pt = Xinit;
iter = 0;
g = vpa(subs(grad, vars, current_pt.'));
d = -g;

while norm(g) > epsilon
    iter = iter + 1;

    if(iter > max_iter)
        disp("Iterated more than max iteration number. Try to run script again. Terminating...")
        return
    end

    hs_points(iter, :) = current_pt;

    subhess = vpa(subs(hess,vars,current_pt.'));
    alpha = -((g.'*d)/(d.'*subhess*d));
    current_pt = vpa(current_pt + alpha.*d);
    g_old = g;
    g = vpa(subs(grad, vars, current_pt.'));

    %update d
    beta = (g.'*(g-g_old))/(d.'*(g-g_old));
    d = -g + beta .* d;

    %plot(current_pt(1), current_pt(2), '+', 'MarkerFaceColor','green','MarkerEdgeColor', 'green');

    if any(abs(current_pt) > xmax) && antidiverge
        disp("Diverged out of the domain! Try to run script again. Terminating...")
        disp("Diverged point:");
        disp(current_pt);
        return
    end
end
exectime = toc;

hs_points(iter+1, :) = current_pt;


disp("------[Hestenes-Stiefel Method]------");
disp("Iterations: " + iter);
disp("Optimum point: ");
disp(current_pt);
disp("Optimum value: ");
disp(vpa(subs(f, vars, current_pt.')));
disp("Gradient: ");
disp(vpa(subs(grad, vars, current_pt.')));
disp("Execution time: ");
disp(exectime);
fprintf("\n\n");

% -----------------------------------------------------------------------
% Polak-Ribiere Method
tic
current_pt = Xinit;
iter = 0;
g = vpa(subs(grad, vars, current_pt.'));
d = -g;

while norm(g) > epsilon
    iter = iter + 1;

    if(iter > max_iter)
        disp("Iterated more than max iteration number. Try to run script again. Terminating...")
        return
    end

    pr_points(iter, :) = current_pt;

    subhess = vpa(subs(hess,vars,current_pt.'));
    alpha = -((g.'*d)/(d.'*subhess*d));
    current_pt = vpa(current_pt + alpha.*d);
    g_old = g;
    g = vpa(subs(grad, vars, current_pt.'));

    %update d
    beta = (g.'*(g-g_old))/(g_old.'*g_old);
    d = -g + beta .* d;

    %plot(current_pt(1), current_pt(2), '+', 'MarkerFaceColor','cyan','MarkerEdgeColor', 'cyan');

    if any(abs(current_pt) > xmax) && antidiverge
        disp("Diverged out of the domain! Try to run script again. Terminating...")
        disp("Diverged point:");
        disp(current_pt);
        return
    end
end
exectime = toc;

pr_points(iter+1, :) = current_pt;

disp("------[Polak-Ribiere Method]------");
disp("Iterations: " + iter);
disp("Optimum point: ");
disp(current_pt);
disp("Optimum value: ");
disp(vpa(subs(f, vars, current_pt.')));
disp("Gradient: ");
disp(vpa(subs(grad, vars, current_pt.')));
disp("Execution time: ");
disp(exectime);
fprintf("\n\n");

% -----------------------------------------------------------------------
% Fletcher-Reeves Method
tic
current_pt = Xinit;
iter = 0;
g = vpa(subs(grad, vars, current_pt.'));
d = -g;

while norm(g) > epsilon
    iter = iter + 1;

    if(iter > max_iter)
        disp("Iterated more than max iteration number. Try to run script again. Terminating...")
        return
    end

    fr_points(iter+1, :) = current_pt;

    subhess = vpa(subs(hess,vars,current_pt.'));
    alpha = -((g.'*d)/(d.'*subhess*d));
    current_pt = vpa(current_pt + alpha.*d);
    g_old = g;
    g = vpa(subs(grad, vars, current_pt.'));

    %update d
    beta = (g.'*g)/(g_old.'*g_old);
    d = -g + beta .* d;

    %plot(current_pt(1), current_pt(2), '+', 'MarkerFaceColor','magenta','MarkerEdgeColor', 'magenta');

    if any(abs(current_pt) > xmax) && antidiverge
        disp("Diverged out of the domain! Try to run script again. Terminating...")
        disp("Diverged point:");
        disp(current_pt);
        return
    end
end
exectime = toc;
fr_points(iter+1, :) = current_pt;

disp("------[Fletcher-Reeves Method]------");
disp("Iterations: " + iter);
disp("Optimum point: ");
disp(current_pt);
disp("Optimum value: ");
disp(vpa(subs(f, vars, current_pt.')));
disp("Gradient: ");
disp(vpa(subs(grad, vars, current_pt.')));disp("Execution time: ");
disp(exectime);
fprintf("\n\n");

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

    if(iter > max_iter)
        disp("Iterated more than max iteration number. Try to run script again. Terminating...")
        return
    end

    qn_points(iter+1, :) = current_pt;

    

    subhess = vpa(subs(hess,vars,current_pt.'));
    alpha = -(g.'*d)/(d.'*subhess*d);
    delta_x = (alpha.*d);
    current_pt = vpa(current_pt + delta_x);
    g_new = vpa(subs(grad, vars, current_pt.'));
    delta_g = g_new - g;
    H = H + (((delta_x - H * delta_g)*(delta_x - H * delta_g).')/(delta_g.' * (delta_x - H * delta_g)));
    d = -H * g_new;
    g = g_new;
    
    if any(abs(current_pt) > xmax)  && antidiverge
        disp("Diverged out of the domain! Try to run script again. Terminating...")
        disp("Diverged point:");
        disp(current_pt);
        return
    end
    
    %plot(current_pt(1), current_pt(2), '+', 'MarkerFaceColor','black','MarkerEdgeColor', 'black');
    
    
    %disp(g);
end
exectime = toc;

qn_points(iter+1, :) = current_pt;

%vpa(subs(f,vars,current_pt.'))
disp("------[Quasi-Newton Method with Rank-One Correction]------");
disp("Iterations: " + iter);
disp("Optimum point: ");
disp(current_pt);
disp("Optimum value: ");
disp(vpa(subs(f, vars, current_pt.')));
disp("Gradient: ");
disp(vpa(subs(grad, vars, current_pt.')));
disp("Execution time: ");
disp(exectime);
fprintf("\n\n");

%plot(current_pt(1), current_pt(2), '+', 'MarkerFaceColor','black','MarkerEdgeColor', 'black');

% Trim preallocated zeros from iteration point data 
nr_points = nonzeros(nr_points);
nr_points_sz = size(nr_points);
nr_points_dim =  (nr_points_sz(1))/2;
nr_points = reshape(nr_points, nr_points_dim, 2);

hs_points = nonzeros(hs_points);
hs_points_sz = size(hs_points);
hs_points_dim =  (hs_points_sz(1))/2;
hs_points = reshape(hs_points, hs_points_dim, 2);

pr_points = nonzeros(pr_points);
pr_points_sz = size(pr_points);
pr_points_dim =  (pr_points_sz(1))/2;
pr_points = reshape(pr_points, pr_points_dim, 2);

fr_points = nonzeros(fr_points);
fr_points_sz = size(fr_points);
fr_points_dim =  (fr_points_sz(1))/2;
fr_points = reshape(fr_points, fr_points_dim, 2);

qn_points = nonzeros(qn_points);
qn_points_sz = size(qn_points);
qn_points_dim =  (qn_points_sz(1))/2;
qn_points = reshape(qn_points, qn_points_dim, 2);

% Plot the iteration paths
plot(nr_points(:,1), nr_points(:,2), '-+', "Color", "#00ff9d", 'LineWidth',1);
plot(hs_points(:,1), hs_points(:,2), '-+', "Color", "#ff3700", 'LineWidth',1);
plot(pr_points(:,1), pr_points(:,2), '-+', "Color", "cyan", 'LineWidth',1);
plot(fr_points(:,1), fr_points(:,2), '-+', "Color", "#FFA500", 'LineWidth',1);
plot(qn_points(:,1), qn_points(:,2), '-+', "Color", "#A020F0", 'LineWidth',1);

plot(Xinit(1), Xinit(2), 'O', 'MarkerFaceColor','magenta','MarkerEdgeColor', 'magenta');
plot(Xstar_pos(1), Xstar_pos(2), 'O', 'MarkerFaceColor','blue','MarkerEdgeColor', 'blue');
plot(Xstar_neg(1), Xstar_neg(2), 'O', 'MarkerFaceColor','red','MarkerEdgeColor', 'red');

hold off

% Turn the iteration path data from 2D -> 3D

nr_vals = arrayfun(f_func, nr_points(:,1), nr_points(:,2));
nr_points = [nr_points nr_vals];

hs_vals = arrayfun(f_func, hs_points(:,1), hs_points(:,2));
hs_points = [hs_points hs_vals];

pr_vals = arrayfun(f_func, pr_points(:,1), pr_points(:,2));
pr_points = [pr_points pr_vals];

fr_vals = arrayfun(f_func, fr_points(:,1), fr_points(:,2));
fr_points = [fr_points fr_vals];

qn_vals = arrayfun(f_func, qn_points(:,1), qn_points(:,2));
qn_points = [qn_points qn_vals];

% 3D path graph
figure(2)
fsurf(f)
hold on

scatter3(Xstar_pos(1),Xstar_pos(2), 0, 72, "blue", "filled")
scatter3(Xstar_neg(1),Xstar_neg(2), 0, 72, "red", "filled")
scatter3(Xinit(1),Xinit(2), f_func(Xinit(1), Xinit(2)), 72, "magenta", "filled")

plot3(nr_points(:,1),nr_points(:,2),nr_points(:,3), '-+', "Color", "#00ff9d",'LineWidth',1);
plot3(hs_points(:,1),hs_points(:,2),hs_points(:,3), '-+', "Color", "#ff3700",'LineWidth',1);
plot3(pr_points(:,1),pr_points(:,2),pr_points(:,3), '-+', "Color", "cyan",'LineWidth',1);
plot3(fr_points(:,1),fr_points(:,2),fr_points(:,3), '-+', "Color", "#FFA500",'LineWidth',1);
plot3(qn_points(:,1),qn_points(:,2),qn_points(:,3), '-+', "Color", "#A020F0",'LineWidth',1);


