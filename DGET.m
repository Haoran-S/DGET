function [Opt, Obj] = DGET(stepsize, PW, y_temp, iter_num, A, n, N, gc,fc, lambda, aalpha, features, labels,  bs, minibatch)
Opt = zeros(iter_num-1,1);
Obj = zeros(iter_num-1,1);
Constraint = zeros(iter_num-1,1);
x = reshape(y_temp(:,1),[n, N]);
grad = zeros(n,N);

grad_old = zeros(n,N);
gradient = zeros(N*n,1);
for ii = 1 : N
    for jj=(ii-1)*bs+1:ii*bs
        gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
    end
    grad(:,ii) = gradient((ii-1)*n+1:ii*n);
end
v_DGET =  grad;
y_DGET  = grad;

for iter  = 2 : iter_num
    for ii = 1 : N
        for jj=(ii-1)*bs+1:ii*bs
            Obj(iter-1,1) = Obj(iter-1,1) + fc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
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
    
    full_grad = mean(gradient_matrix,2);
    x_vec = reshape(x,[N*n,1]);
    Constraint(iter-1,1) =  norm(A*x_vec(:,1))^2;
    Opt(iter-1,1) = norm(full_grad)^2+1/N *Constraint(iter-1,1);
    
    
    % start
    alpha = stepsize;
    
    % Update x
    x_old = x;
    x = x*PW - alpha * y_DGET;
    
    % Update v
    v_DGET_old = v_DGET;
    if(mod(iter-2, bs)==0)
        gradient = zeros(N*n,1);
        for ii = 1 : N
            for jj=(ii-1)*bs+1:ii*bs
                gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
            end
            grad(:,ii) = gradient((ii-1)*n+1:ii*n);
        end
        v_DGET = grad;
    else
        sample = randi(bs,1,minibatch);
        gradient = zeros(N*n,1);
        for ii = 1 : N
            for jj=(ii-1)*bs + sample 
                gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
            end
            grad(:,ii) = gradient((ii-1)*n+1:ii*n);
        end
        
        gradient = zeros(N*n,1);
        for ii = 1 : N
            for jj=(ii-1)*bs + sample 
                gradient((ii-1)*n+1:ii*n) = gradient((ii-1)*n+1:ii*n) + gc(x_old(:,ii),lambda,aalpha, features(:,jj), labels(jj),bs, N);
            end
            grad_old(:,ii) = gradient((ii-1)*n+1:ii*n);
        end
        v_DGET = v_DGET + grad - grad_old;
    end
    
    % Update y
    y_DGET = y_DGET*PW + v_DGET - v_DGET_old;
    
    
    
    
end
end