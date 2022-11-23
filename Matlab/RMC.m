function [Err] = RMC(u,v,L,n,w,w_o)
    w = zeros(n,1);
    d_k = w_o'*u+v;
    P = eye(n,n);
    P = 1*P;
    Err = ones(1,L);
    lambda = 1;
    sigma = 1;
    for k = 1:L
        e_k = d_k(:,k)-w'*u(:,k);
        K = P*u(:,k)/(lambda*exp((e_k.*e_k)/(2*sigma*sigma))+u(:,k)'*P*u(:,k));
        P = 1/lambda*(P-K*u(:,k)'*P);
        w=w+K*e_k;
        Err(k) = norm(w-w_o);
    end
end