function [Opt_GNSD, Obj_GNSD] = GNSD(stepsize, PW, x_temp, iter_num, A, n, N, gc, fc,lambda, aalpha, features, labels,  bs, minibatch)
Opt_GNSD = zeros(iter_num-1,1);
Obj_GNSD = zeros(iter_num-1,1);
Constraint = zeros(iter_num-1,1);
x = reshape(x_temp(:,1),[n, N]);
grad = zeros(n,N);
time = zeros(iter_num,1);
for iter  = 2 : iter_num
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            Obj_GNSD(iter-1,1) = Obj_GNSD(iter-1,1) + fc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
    end
    % calculating the opt-gap
    gradient = zeros(N*n,1);
    gradient_matrix = zeros(n,N);
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        gradient_matrix(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    time(iter, 1) = time(iter-1, 1) + toc;
    full_grad = mean(gradient_matrix,2);
    x_vec = reshape(x,[N*n,1]);
    Constraint(iter-1,1) =  norm(A*x_vec(:,1))^2;
    Opt_GNSD(iter-1,1) = norm(full_grad)^2+1/N *Constraint(iter-1,1);
    
   
    alpha = stepsize;
    
    gradient = zeros(N*n,1);
    sample = randi(bs,1,minibatch);
    for ii = 1 : N
        for jj=(ii-1)*bs + sample
            gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
        end
        grad(:,ii) = gradient((ii-1)*n+1:ii*n);
    end
    
    if(iter==2)
        x_mid =  x -  alpha * grad;
    else
        x_mid = 2 * x * PW - x_old * PW * PW -  alpha * (grad -  grad_old);
    end

    grad_old = grad;
    x_old = x;
    x = x_mid;
end
end