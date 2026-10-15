function [rick]=ricker(np,fr,dt)
npt=np*dt;
t=(-npt/2):dt:npt/2;
rick1=(1-t .*t * fr^2 *pi^2  ) .*exp(- t.^2 * pi^2 * fr^2 ) ;
rick=rick1(round(np/2)-round(1/fr/dt)+1:np);
l=length(rick);
% plot([0:l-1]*dt,rick);xlabel('Time (s)');
% title([num2str(fr),' Hz Ricker Wavelet']);pause(1);
