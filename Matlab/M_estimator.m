function [Err] = M_estimator(u,v,L,n,w,w_o)
    d_k = w'*u+v;
    lr = 0.01;
    Err = ones(1,L);
    w_k1 = w_o;
    w_k_1 = eye(n,n);
    s = 1;
    for k=1:L
        e_k = d_k(k)-w_k1'*u(:,k);
        w_k1 = w_k1+(inv((u(:,k)'*w_k_1*u(:,k)))*u(:,k)'*w_k_1*(e_k))';
        Err(k) = norm(w_k1-w);
        w_k_1 = eye(n,n)*sum(w_k1'*u(:,k))/50;
%         for i = 1:n
%             w_k_1(i,i) = d_k(i)-sum(w1'*u(i,k));
%         end
        % w1 = w_k1;
    end
end