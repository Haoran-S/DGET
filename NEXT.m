function [Opt_NEXT, Obj_NEXT] = NEXT(PW, y_temp,  iter_num,  A, n,N,gc,fc,lambda,aalpha, features, labels,bs)
x_next_vec = y_temp(:,1);
Opt_NEXT = zeros(iter_num-1,1);
Obj_NEXT = zeros(iter_num-1,1);
Constraint_NEXT = zeros(iter_num-1,1);
x_next = reshape(x_next_vec,[n, N]);
x_tilde_next = zeros(n,N);
y_next = zeros(n,N);
pi_tilde_next = zeros(n,N);
grad_next = zeros(n,N);
time = zeros(iter_num,1);

gradient = zeros(N*n,1);
for ii = 1 : N
    for jj=(ii-1)*bs+1:ii*bs
        gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x_next_vec((ii-1)*n+1:ii*n,1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
    end
    grad_next(:,ii) = gradient((ii-1)*n+1:ii*n);
    y_next(:,ii) = grad_next(:,ii);
    pi_tilde_next(:,ii) = N*y_next(:,ii) - grad_next(:,ii);
end


tau = ones(N,1);
alpha = 0.001;
for iter  = 2 : iter_num
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            Obj_NEXT(iter-1,1) = Obj_NEXT(iter-1,1) + fc(x_next_vec((ii-1)*n+1:ii*n,1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
    end
    
    % calculating the opt-gap
    full_grad = mean(grad_next,2);
    Constraint_NEXT(iter-1,1) =  norm(A*x_next_vec(:,1))^2;
    Opt_NEXT(iter-1,1) = norm(full_grad)^2  +1/N *Constraint_NEXT(iter-1,1);
    
    %%%% Local SCA optimization  %%%%
    %%%% a) update x             %%%%
    gradient = zeros(N*n,1);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x_next_vec((ii-1)*n+1:ii*n,1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        grad_next(:,ii) = gradient((ii-1)*n+1:ii*n);
        x_tilde_next(:,ii) = (-grad_next(:,ii) + tau(ii)*x_next(:,ii) - pi_tilde_next(:,ii))./tau(ii);
    end
    
    %%% b) update z               %%%
    alpha = alpha * (1- 0.01 * alpha);
    z_next = x_next + alpha*(x_tilde_next - x_next);
    %%% Consensus update          %%%
    x_next = z_next*PW; % take the average
    grad_next_old = grad_next;
    x_next_vec = reshape(x_next,[N*n,1]);
    gradient = zeros(N*n,1);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x_next_vec((ii-1)*n+1:ii*n,1),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        grad_next(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    y_next = y_next*PW + grad_next - grad_next_old;
    pi_tilde_next = N*y_next-grad_next;
    
    time(iter, 1) = time(iter-1, 1) + toc;
    
    
    
end
