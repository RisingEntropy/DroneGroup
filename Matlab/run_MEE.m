close all;
clear all;
% 0.25
n = 10;
L = 1000;
EE1 = ones(1,L);
EE2 = ones(1,L);
EE3 = ones(1,L);

for i = 1:100
    u_k = randn(n,L);
%     v_k = randn(1,L);

    L=1000;
    noise1 = (randn(1,L))*0.1+5;
    noise2 = (randn(1,L))*5-6;
    noisee = noise1;
    bool_map = noisee>0.09;
    noisee = noisee.*(1-bool_map)+noise2.*bool_map;

    w = randn(n,1);
    w1 = randn(n,1);
    E1 = DLMS(u_k,noisee,L,n,w,w1);
    E2 = MCC(u_k,noisee,L,n,w,w1);
    E3 = MEE(u_k,noisee,L,n,w,w1);
    EE1 = EE1 + E1;
    EE2 = EE2 + E2;
    EE3 = EE3 + E3;
end
EE1 = EE1/100;
EE2 = EE2/100;
EE3 = EE3/100;
figure,hold on
plot(20*log10(EE1),'r');
plot(20*log10(EE2),'g');
plot(20*log10(EE3),'b');
legend("LMS","MCC","MEE");
