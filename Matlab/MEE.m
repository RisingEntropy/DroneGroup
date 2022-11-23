function [Err] = MEE(u,v,L,n,w,w_o)
    w1 = w_o;
    d_k = w'*u+v;
    lr = 8;
    Err = ones(1,L);
    window_Len = 5;
    sigma = 4;
    e_k_arr = [];
    for k=1:L
        e_k_arr = [e_k_arr,d_k(:,k)-w1'*u(:,k)];
        w_k1 = w1;
        for i = max(1,k-window_Len+1):k
            for j = max(1,k-window_Len+1):k
                w_k1 = w_k1+lr*((1)/(window_Len^2*sigma^2))*G_sigma(e_k_arr(i)-e_k_arr(j),sigma)*(e_k_arr(i)-e_k_arr(j))*(u(:,i)-u(:,j));
            end
        end
        Err(k) = norm(w_k1-w);
        w1 = w_k1;
    end
end