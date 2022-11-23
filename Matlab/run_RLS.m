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
%     v_k = randn(1,L);

    L=1000;
    noise1 = (randn(1,L))*0.1;
    noise2 = (randn(1,L))*5;
    noisee = noise1;
    bool_map = noisee>0.09;
    noisee = noisee.*(1-bool_map)+noise2.*bool_map;

    w = randn(n,1);
    w1 = randn(n,1);
    E4 = DRLS(u_k,noisee,L,n,w,w1);
    E5 = RMC(u_k,noisee,L,n,w,w1);
    wk3 = inv(u_k*u_k')*u_k*(w'*u_k+noisee)';
    E3 = ones(1,L)*norm(wk3-w);
    EE3=EE3+E3;
    EE4=EE4+E4;
    EE5=EE5+E5;
end
EE3=EE3/100;
EE4=EE4/100;
EE5=EE5/100;
figure,hold on
plot(20*log10(EE4),'r');
plot(20*log10(EE5),'g');
plot(20*log10(EE3),'b');
% plot(20*log10(EE4),'k');
plot(20*log10(EE5),'c');
% legend('NLMS','LMS',"Wiener","RLS","VSS");
legend("RLS","RMC","Wiener");
