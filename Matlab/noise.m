close all;
clear all;
L=1000;
noise1 = (randn(1,L))*0.1;
noise2 = (randn(1,L))*5;
noisee = noise1;
bool_map = noisee>0.09;
noisee = noisee.*(1-bool_map)+noise2.*bool_map;
plot(noisee);

