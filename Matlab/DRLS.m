function [Err] = DRLS(u,v,L,n,w,w_o)
    w = zeros(n,1);
    d_k = w_o'*u+v;
    P = eye(n,n);
    P = 1*P;
    Err = ones(1,L);
    lambda = 1;
    for k = 1:L
        e = d_k(:,k)-w'*u(:,k);
        K = P*u(:,k)/(lambda+u(:,k)'*P*u(:,k));
        P = 1/lambda*(P-K*u(:,k)'*P);
        w=w+K*e;
        Err(k) = norm(w-w_o);
    end
end