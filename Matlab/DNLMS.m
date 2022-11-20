function [Err] = DNLMS(u,v,L,n,w,w_o)
    d_k = w'*u+v;
    lr = 0.1;
    Err = ones(1,L);
    w1 = w_o;
    for k=1:L
        e_k = d_k(:,k)-w1'*u(:,k);
        w_k1 = w1+lr*e_k*(u(:,k)/(u(:,k)'*u(:,k)));
        Err(k) = norm(w_k1-w);
        w1 = w_k1;
    end
end