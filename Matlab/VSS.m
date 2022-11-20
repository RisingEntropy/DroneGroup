function [Err] = VSS(u,v,L,n,w,w_o)
    d_k = w'*u+v;
    lr = 0.01;
    Err = ones(1,L);
    w1 = w_o;
    alpha = 0.7;
    gama = 0.9;
    mu_max = 0.1;
    mu_min = 0.001;
    mu = mu_max;
    for k=1:L
        e_k = d_k(:,k)-w1'*u(:,k);
        w_k1 = w1+lr*e_k*u(:,k);
        Err(k) = norm(w_k1-w);
        mu = alpha*mu+gama*Err(k);
        mu = min(mu,mu_max);
        mu = max(mu,mu_min);
        w1 = w_k1;
    end
end