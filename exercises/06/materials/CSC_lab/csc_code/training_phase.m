%% Do the convolutional sparse coding for denoising -- training phase
% solve both filter d and sparse coefficients z
% Zhaolun Liu on 2018.10.28
% zhaolun.liu@kaust.edu.sa

clear;
close all;
tic();
% 
%
% *INPUT PARAMETERS* :
% x                 Input data with size of n1*n2*n3, where n1*n2 is the  image size and n3 is the number of the images
load b.mat % load 5 common shot gathers (portions)
b1=b(1:100,:,1:1); % chose the first one to learing the filters
x=local_contrast_normalization(b1,21); % do local contrast normalization

n3=size(x,3); % zero mean
for img=1:n3
    tmp=x(:,:,img);
    tmp=tmp-mean(tmp(:));
    x(:,:,img)=tmp;
end

% figure;
% imagesc(x(:,:,1));title('Original CSG');colormap gray; %plot the images x
% filter_size       Size of filters: n1*n2*n3, where n1*n2 is the size for the single filter and n3 is the number of the filters
filter_size = [21,11,3];
fprintf('Doing sparse coding filter learning for k = %d [%d x %d] filters.\n\n', filter_size(3), filter_size(1), filter_size(2) )
% ambda_residual   The coefficient before the first term 1/2||b- \sum_i d_i*z_i||^2_2
lambda_residual = 1.0;
% beta            The coefficient before the second term \sum_i ||z_i||_1
beta = 1.0;
% max_it            The maximum iteration number
max_it = [80];
% tol               Tolerenace to stop the iteration
tol = 1e-3;
% verbose_admm      Optim options. Two types of output information: 'breif' or 'all' 
verbose_admm = 'all';
% 
%
% *OUTPUT PARAMETERS* :
% > d                 Final filters
% > z                 Final coefficients
% > Dz                Reconstructed image with extended boundaries Dz=\sum d_i*z_i
% > obj               The final misfit of the objective function
% > iterations        Including information for each iteration
%                       > iterations.obj_vals_d: misfit for updating the
%                            filters for each iteration
%                       > iterations.obj_vals_z: misfit for updating the
%                            coefficients for each iteration
%                       > iterations.tim_vals: elapsed time for each iter
%                       > iterations.it_vals: Filters for each iteration
[ d, z, Dz, obj, iterations]  = admm_learn_conv2D_weighted_test(x, filter_size, lambda_residual, beta, max_it, tol, verbose_admm, []);
prefix = 'ours';
save('filters.mat', 'd', 'z', 'Dz', 'obj','filter_size', 'iterations');
tt = toc

%% Show filters result
% Debug options
verbose = 'brief';

figure();    
pd = 1;
sqr_k = ceil(sqrt(size(d,3)));
%     filter_size(2)=21;
d_disp = zeros( sqr_k * [filter_size(1) + pd, filter_size(2) + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (filter_size(1) + pd) + pd + (1:filter_size(1)) , mod(j,sqr_k) * (filter_size(2) + pd) + pd + (1:filter_size(2)) ) = d(:,:,j + 1); 
end
imagesc(d_disp), colormap gray, 
axis image, colorbar, title('Final filter estimate');

%% Display filters, feature maps and their convolution results
% load('filters_ours_obj467.mat')
psf_radius = floor(size(d,1)/2);
psf_radius_2=floor(size(d,2)/2);
fontsize=10;
for ifilter=1:size(z,3)
    figure;

    %plot filter
    pos1 = [0.05 0.5 0.05 0.1];
    subplot('Position',pos1)
    imagesc(d(:,:,ifilter));
    annotation('textbox',[0.12 0.5 0.05 0.1],'String','*','EdgeColor','none')
    annotation('textbox',[0.52 0.5 0.05 0.1],'String','=','EdgeColor','none')
    title(['Filter $\bf d_',num2str(ifilter),'$'],'interpreter','latex');
    colormap gray;
    set(gca,'FontSize',fontsize);

    %plot feature map
    pos2 = [0.2 0.4 0.3 0.3];
    subplot('Position',pos2)
    d_disp=abs(z(:,:,ifilter));
    d_disp(abs(d_disp(:))<0.002*max(abs(z(:))))=0;
    % color_array_define; spy(d_disp(psf_radius:end-psf_radius,psf_radius_2:end-psf_radius_2),10,co{ifilter})
    spy_lz(d_disp(psf_radius:end-psf_radius,psf_radius_2:end-psf_radius_2),3,'k')
    title(['Feature map $\bf z_',num2str(ifilter),'$'],'interpreter','latex');
    set(gca,'FontSize',fontsize);
    
    %plot convolution result
    pos3 = [0.6 0.4 0.3 0.3];
    subplot('Position',pos3)
    tmp=conv2(z(:,:,ifilter),d(:,:,ifilter));
    imagesc(tmp(2*psf_radius:end-2*psf_radius,2*psf_radius_2:end-2*psf_radius_2)); colormap gray;
    title(['$\bf b_',num2str(ifilter),'* z_',num2str(ifilter),'$'],'interpreter','latex');
    set(gca,'FontSize',fontsize);
end

