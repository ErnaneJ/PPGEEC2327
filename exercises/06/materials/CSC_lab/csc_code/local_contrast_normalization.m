%% The Local Constrast Normalization module 
% performs local subtraction and division normalizations, 
% enforcing a sort of local competition between adjacent features in a feature map, and between 
% features at the same spatial location in different feature maps.
% More introduction at http://yeephycho.github.io/2016/08/03/Normalizations-in-neural-networks/
% Zhaolun Liu on 2018.10.28
% zhaolun.liu@kaust.edu.sa

function [b_filter]=local_contrast_normalization(b,nk)
k = fspecial('gaussian',[nk nk],3*1.591);
nb=size(b,3);
b_filter=b;
for ib=1:nb
dim=b(:,:,ib);
lmn=rconv2(dim,k);
lmnsq=rconv2(dim.^2,k);
lvar=lmnsq-lmn.^2;
lvar(lvar<0) = 0; % avoid numerical problems
lstd = sqrt(lvar);
q=sort(lstd(:));
lq = round(length(q)/2);
th = q(lq);
if(th==0)
    q = nonzeros(q);
    if(~isempty(q))
    lq = round(length(q)/2);
    th = q(lq);
    else
        th = 0;
    end
end
lstd(lstd<=th) = th;
lstd(lstd(:)==0) = eps;
dim = dim - lmn;
dim = dim ./ lstd;
b_filter(:,:,ib)=dim(:,:);
end


function c = rconv2(large,small)

ctr = 0;
ly = size(large,1);
lx = size(large,2);
sy = size(small,1);
sx = size(small,2);

%% These values are one less than the index of the small mtx that falls on 
%% the border pixel of the large matrix when computing the first 
%% convolution response sample:
sy2 = floor((sy+ctr-1)/2);
sx2 = floor((sx+ctr-1)/2);

% pad with reflected copies
clarge = [ 
    large(sy-sy2:-1:2,sx-sx2:-1:2), large(sy-sy2:-1:2,:), ...
	large(sy-sy2:-1:2,lx-1:-1:lx-sx2); ...
    large(:,sx-sx2:-1:2),    large,   large(:,lx-1:-1:lx-sx2); ...
    large(ly-1:-1:ly-sy2,sx-sx2:-1:2), ...
      large(ly-1:-1:ly-sy2,:), ...
      large(ly-1:-1:ly-sy2,lx-1:-1:lx-sx2) ];

  
%   keyboard
  
c = conv2(clarge,small,'valid');