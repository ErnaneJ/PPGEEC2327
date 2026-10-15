function D_out=Normalize(D_in)
%  Copyright (C) 2010 Center for Subsurface Imaging and Fluid Modeling (CSIM),
%  King Abdullah University of Science and Technology, All rights reserved.
%
%  author:   Xin Wang
%  email:    xin.wang@kaust.edu.sa
%  date:     Feb 15 , 2013
%
%  NORMALIZE: Normalizing
%
%  d_out=Normalize(D_in)
%
%  IN   d_in   : input data
%  OUT  d_out  : output normalized data
%
%  Example
%    load segsalt2D_vel_nz150_nx645_dx5m;
%    vel1=Normalize(vel);
%    figure;subplot(211);imagesc(vel);colorbar;
%    subplot(212); imagesc(vel1); colorbar;
D_out=D_in/max(abs(D_in(:)));
end