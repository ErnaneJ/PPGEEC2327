%% Do the convolutional sparse coding for denoising -- denoising phase
% solve sparse coefficients z with learned filter d
% Zhaolun Liu on 2018.10.28
% zhaolun.liu@kaust.edu.sa

close all;
clear;
% 
%%% !!!!!!!!!!!!!!!!!!!!load filter !!!!!!need to change your trained filter
filters=load('filters.mat');
d = filters.d;
%
% *INPUT PARAMETERS* :
% load the 2-th csg
% x                 Input data with size of n1*n2*n3, where n1*n2 is the  image size and n3 is the number of the images
load b.mat % load 5 common shot gathers (portions)
b=b(1:100,:,2); % chose the first one to learing the filters
x=local_contrast_normalization(b,21); % do local contrast normalization

%%% add noise to the csg 
n3=size(x,3); % zero mean
for img=1:n3
    tmp=x(:,:,img);
    tmp=tmp-mean(tmp(:));
    x(:,:,img)=tmp;
    x_noise(:,:,img)=tmp+ 0.3 * randn(size(tmp));%control noise level

end


%%% other parameters
% ambda_residual   The coefficient before the first term 1/2||b- \sum_i d_i*z_i||^2_2
lambda_residual = 1.0;
% ambda            The coefficient before the second term \sum_i ||z_i||_1
lambda = 0.5;
% max_it            The maximum iteration number
max_it = [80];
% tol               Tolerenace to stop the iteration
tol = 1e-3;
% verbose_admm      Optim options. Two types of output information: 'breif' or 'all' 
verbose_admm = 'all';
% mask
MtM = zeros(size(x_noise));
MtM(:)=1;


% 
%
% *OUTPUT PARAMETERS* :
% > z                 Final coefficients
% > sig_rec           Reconstructed image 
[z, sig_rec] = admm_solve_conv2D_weighted_sparse_reconstruction(x_noise, d, MtM, lambda_residual, lambda, max_it, 1e-3, x_noise, verbose_admm); 

sig_to_noise_ratio_before=snr_test(x(:),x_noise(:)-x(:));
sig_to_noise_ratio_after=snr_test(x(:),sig_rec(:)-x(:));
figure;
subplot(2,2,1);imagesc(x);title('a) Original CSG');colormap gray%plot the images x
subplot(2,2,2);imagesc(x_noise);title(['b) CSG with noise(SNR=',num2str(sig_to_noise_ratio_before),')']);colormap gray
subplot(2,2,3);imagesc(sig_rec);title(['c) Denoised CSG (SNR=',num2str(sig_to_noise_ratio_after),')']);colormap gray%plot the images x
subplot(2,2,4);imagesc(x_noise-sig_rec);title('d) Diff');colormap gray
%% Show filter result
showall_flag='no';
if  strcmp(showall_flag, 'yes')
    % Debug options
    verbose = 'brief';
    prefix = 'ours';
    figure();   
    filter_size=filters.filter_size;
    pd = 1;
    sqr_k = ceil(sqrt(size(d,3)));
    %     filter_size(2)=21;
    d_disp = zeros( sqr_k * [filter_size(1) + pd, filter_size(2) + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_disp( floor(j/sqr_k) * (filter_size(1) + pd) + pd + (1:filter_size(1)) , mod(j,sqr_k) * (filter_size(2) + pd) + pd + (1:filter_size(2)) ) = d(:,:,j + 1); 
    end
    imagesc(d_disp), colormap gray, 
    %     axis image, colorbar, title('Final filter estimate');

    % Display filters, feature maps and their convolution results
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
        title('Filter $\bf d_k$','interpreter','latex');
        colormap gray;
        set(gca,'FontSize',fontsize);

        %plot feature map
        pos2 = [0.2 0.4 0.3 0.3];
        subplot('Position',pos2)
        d_disp=abs(z(:,:,ifilter));
        d_disp(abs(d_disp(:))<0.002*max(abs(z(:))))=0;
        % color_array_define; spy(d_disp(psf_radius:end-psf_radius,psf_radius_2:end-psf_radius_2),10,co{ifilter})
        spy_lz(d_disp(psf_radius:end-psf_radius,psf_radius_2:end-psf_radius_2),3,'k')
        title('Feature map $\bf z_k$','interpreter','latex');
        set(gca,'FontSize',fontsize);

        %plot convolution result
        pos3 = [0.6 0.4 0.3 0.3];
        subplot('Position',pos3)
        tmp=conv2(z(:,:,ifilter),d(:,:,ifilter));
        imagesc(tmp(2*psf_radius:end-2*psf_radius,2*psf_radius_2:end-2*psf_radius_2)); colormap gray;
        title('$\bf b_k* z_k$','interpreter','latex');
        set(gca,'FontSize',fontsize);
    end
end

