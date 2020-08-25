clc; clear; close all;
tic
rng('default')
file_name = 'a9a';
prob = 'line_log';

% % % --- logistic regression % % % 
gc = @(x,lambda,alpha,v,y,bs, M) 1/(bs*M) * ((-y*v) / (1+exp(x'*v)) -(exp(x'*v)* v* (-1 + y))/(1 + exp(x'*v)) )  + 1/(bs*M) * 2*lambda*x ./ (1+x.^2).^2; % gradient
fc = @(x,lambda,alpha,v,y,bs, M) 1/(bs*M) * (-y * log(1 / (1+exp(-x'*v))) - (1-y) * log(1 - 1 / (1+exp(-x'*v))) )   + 1/(bs*M) * lambda * sum(x.^2 ./ (1+x.^2)); % objective

% % % --- linear regression % % % 
% gc = @(x,lambda,alpha,v,y,bs, M) 1/(bs*M) *  -(v * (-x' *v + y))/(1 + 1/2 * (-x'*v + y)^2); % gradient
% fc = @(x,lambda,alpha,v,y,bs, M) 1/(bs*M) *  log((y - x'*v)^2/2 + 1); % objective


% % Parameters
n          = 123;   % problem dimention
batch_size = 3256;  % batch size
minibatch  = 64;
epoch_length = batch_size / minibatch;
nodes_num  = 10;    % number of agents in the network
K = batch_size * nodes_num; % number of data points
repeat_num = 1;     % number of trials
epoch_num  = 12;    % number of iterations per trial
radius     = 0.5;

function_lambda = 0.01;
function_aalpha = 1;


load(sprintf('%s_train.mat', file_name))
labels = cast(labels,'double');
labels(labels==-1) = 0;

rand_index = randperm(size(labels, 1));
features = features(rand_index, :);
labels = labels(rand_index, :);
features = features(1:K, :)';
labels = labels(1:K, :);
stepsize = 0.01;


% % Algorithms
for repeat_index = 1 : repeat_num
    disp(repeat_index);
    [Adj, degree, num_of_edge,A,B,D,Lm,edge_index, eig_Lm,min_eig_Lm,WW,LN,L_hat,eig_L_hat,min_eig_L_hat] = Generate_Graph(nodes_num,radius,n);
    x_initial = zeros(nodes_num*n,epoch_num);
    x_initial(:,1) = randn(nodes_num*n,1);
    % Metropolis-weight matrix
    PW = zeros(nodes_num, nodes_num);
    for ii = 1 : nodes_num
        for jj = ii+1 : nodes_num
            if Adj(ii,jj) == 1
                PW(ii,jj) =  1.0/(1+max(degree(ii), degree(jj)));
                PW(jj,ii) = PW(ii,jj);
            end
        end
        PW(ii,ii) = 1-sum(PW(ii,:));
    end
 
    [Opt_NEXT(:,repeat_index), Obj_NEXT(:,repeat_index)] = NEXT(PW, x_initial,  round(epoch_num * epoch_length),   A,  n,nodes_num,gc,fc,function_lambda,function_aalpha, features, labels, batch_size);
    [Opt_DGET(:,repeat_index), Obj_DGET(:,repeat_index)] = DGET(stepsize, PW, x_initial,  round(epoch_num * epoch_length), A, n,nodes_num,gc,fc,function_lambda,function_aalpha, features, labels, batch_size, minibatch);
    [Opt_GNSD(:,repeat_index), Obj_GNSD(:,repeat_index)] = GNSD(stepsize, PW, x_initial,  round(epoch_num * epoch_length), A, n,nodes_num,gc,fc,function_lambda,function_aalpha, features, labels,batch_size, minibatch);
    [Opt_PSGD(:,repeat_index), Obj_PSGD(:,repeat_index)] = PSGD(stepsize, PW, x_initial,  round(epoch_num * epoch_length), A, n,nodes_num,gc,fc, function_lambda,function_aalpha, features, labels,batch_size, minibatch);
end

% % plot the results
linewidth = 1;
fontsize = 11;
x_axis = (0:size(mean(Obj_NEXT,2))-1)./2;
x_axis_scale = (0:size(mean(Opt_GNSD,2))-1)./epoch_length;

figure(1)
semilogy(x_axis, mean(Opt_NEXT,2),'linestyle', '-.','linewidth',linewidth,  'Marker', 'o', 'MarkerIndices',1:round(epoch_num/7):length(x_axis));hold on;
semilogy(x_axis_scale, mean(Opt_PSGD,2),'linestyle', ':','linewidth',linewidth,  'Marker', '+', 'MarkerIndices',1:round(length(x_axis_scale)/10):length(x_axis_scale));hold on;
semilogy(x_axis_scale, mean(Opt_GNSD,2),'linestyle', '--','linewidth',linewidth,   'Marker', 'x', 'MarkerIndices',1:round(length(x_axis_scale)/9):length(x_axis_scale));hold on;
semilogy(x_axis_scale, mean(Opt_DGET,2),'linestyle', '-','linewidth',linewidth,  'Marker', 's', 'MarkerIndices',1:round(length(x_axis_scale)/8):length(x_axis_scale));hold on;
xlim([0,epoch_num-2]);
le = legend( 'NEXT', 'PSGD', 'GNSD', 'DGET');
xl = xlabel('Epoch','FontSize',fontsize);
yl = ylabel('Optimality Gap h^*','FontSize',fontsize);
savefig(sprintf('figure_%s_%s_bs%d_ep%d_%f_opt.fig',file_name, prob, batch_size, epoch_num, stepsize));


figure(2)
semilogy(x_axis, mean(Obj_NEXT,2),'linestyle', '-.','linewidth',linewidth, 'Marker', 'o', 'MarkerIndices',1:round(epoch_num/7):length(x_axis));hold on;
semilogy(x_axis_scale, mean(Obj_PSGD,2),'linestyle', ':','linewidth',linewidth,  'Marker', '+', 'MarkerIndices',1:round(length(x_axis_scale)/10):length(x_axis_scale));hold on;
semilogy(x_axis_scale, mean(Obj_GNSD,2),'linestyle', '--','linewidth',linewidth,  'Marker', 'x', 'MarkerIndices',1:round(length(x_axis_scale)/9):length(x_axis_scale));hold on;
semilogy(x_axis_scale, mean(Obj_DGET,2),'linestyle', '-','linewidth',linewidth,  'Marker', 's', 'MarkerIndices',1:round(length(x_axis_scale)/8):length(x_axis_scale));hold on;
xlim([0,epoch_num-2]);
le = legend( 'NEXT', 'PSGD', 'GNSD', 'DGET');
xl = xlabel('Epoch','FontSize',fontsize);
yl = ylabel('Objective Value','FontSize',fontsize);
savefig(sprintf('figure_%s_%s_bs%d_ep%d_%f_loss.fig',file_name, prob, batch_size, epoch_num, stepsize));

figure(3)
semilogy(mean(Opt_NEXT,2),'linestyle', '-.','linewidth',linewidth, 'Marker', 'o', 'MarkerIndices',1:round(length(x_axis)/20):length(x_axis));hold on;
semilogy(mean(Opt_PSGD,2),'linestyle', ':','linewidth',linewidth,  'Marker', '+', 'MarkerIndices',1:round(length(x_axis_scale)/10):length(x_axis_scale));hold on;
semilogy(mean(Opt_GNSD,2),'linestyle', '--','linewidth',linewidth,  'Marker', 'x', 'MarkerIndices',1:round(length(x_axis_scale)/9):length(x_axis_scale));hold on;
semilogy(mean(Opt_DGET,2),'linestyle', '-','linewidth',linewidth,  'Marker', 's', 'MarkerIndices',1:round(length(x_axis_scale)/8):length(x_axis_scale));hold on;
le = legend( 'NEXT', 'PSGD', 'GNSD', 'DGET');
xl = xlabel('Communication Rounds','FontSize',fontsize);
yl = ylabel('Optimality Gap','FontSize',fontsize);
savefig(sprintf('figure_%s_%s_bs%d_ep%d_%f_opt_comm.fig',file_name, prob, batch_size, epoch_num, stepsize));

figure(4)
semilogy(mean(Obj_NEXT,2),'linestyle', '-.','linewidth',linewidth,  'Marker', 'o', 'MarkerIndices',1:round(length(x_axis)/20):length(x_axis));hold on;
semilogy(mean(Obj_PSGD,2),'linestyle', ':','linewidth',linewidth,  'Marker', '+', 'MarkerIndices',1:round(length(x_axis_scale)/10):length(x_axis_scale));hold on;
semilogy(mean(Obj_GNSD,2),'linestyle', '--','linewidth',linewidth,  'Marker', 'x', 'MarkerIndices',1:round(length(x_axis_scale)/9):length(x_axis_scale));hold on;
semilogy(mean(Obj_DGET,2),'linestyle', '-','linewidth',linewidth,  'Marker', 's', 'MarkerIndices',1:round(length(x_axis_scale)/8):length(x_axis_scale));hold on;
le = legend( 'NEXT', 'PSGD', 'GNSD', 'DGET');
xl = xlabel('Communication Rounds','FontSize',fontsize);
yl = ylabel('Objective Value','FontSize',fontsize);
savefig(sprintf('figure_%s_%s_bs%d_ep%d_%f_loss_comm.fig',file_name, prob, batch_size, epoch_num, stepsize));
