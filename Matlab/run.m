close all;
clear all;
% 0.25
n = 10;
L = 1000;
EE1 = ones(1,L);
EE2 = ones(1,L);
EE3 = ones(1,L);
EE4 = ones(1,L);
EE5 = ones(1,L);
for i = 1:100
    u_k = randn(n,L);
    v_k = randn(1,L);
    w = randn(n,1);
    w1 = randn(n,1);
    E1 = DNLMS(u_k,v_k,L,n,w,w1);
    E2 = DLMS(u_k,v_k,L,n,w,w1);
    E4 = DRLS(u_k,v_k,L,n,w,w1);
    E5 = E_estimator(u_k,v_k,L,n,w,w1);
    wk3 = inv(u_k*u_k')*u_k*(w'*u_k+v_k)';
    E3 = ones(1,L)*norm(wk3-w);
    EE1=EE1+E1;
    EE2=EE2+E2;
    EE3=EE3+E3;
    EE4=EE4+E4;
    EE5=EE5+E5;
end
EE1=EE1/100;
EE2=EE2/100;
EE3=EE3/100;
EE4=EE4/100;
EE5=EE5/100;
figure,hold on
plot(EE1,'r');
plot(EE2,'g');
plot(EE3,'b');
plot(EE4,'k');
plot(EE5,'c');
legend('NLMS','LMS',"Wiener","RLS","E_estimator");
