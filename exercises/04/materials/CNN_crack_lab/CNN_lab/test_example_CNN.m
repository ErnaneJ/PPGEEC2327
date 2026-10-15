clear all,close all

addpath ./CNN
addpath ./util
addpath ./data/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Inputs                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntrain=200; % The number of training set

wdx=8; 
wdz=8;
image_x=wdx*2+1; % Original Image Patch Size X
image_z=wdz*2+1; % Original Image Patch Size Z
% Intepolate image to make sure size before pooling is even
image_x_new=32;  % Intepolated Image Patch Size X
image_z_new=32;  % Intepolated Image Patch Size Z

opts.alpha = 1;   % Learning Rate
opts.batchsize = 20;  % Minibatch Size
opts.numepochs = 10;  % Iteration Number

%%%%  CNN structure %%%%%%%
cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer   
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer   
};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%              Step 1 Prepare test and training set            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load image_input.mat 

%%%%%%%%%%%% Load Prepared training set%%%%%%%%%%%%% 
load train_new.mat

% %%%%%%%%%%% Load Prepared training set%%%%%%%%%%%%% 
% % Pick Train Examples
% % Pick Crack points
% figure(1)
% imagesc(image_input);
% title('Pick Crack Points','fontsize',22);
% train.label(1:ntrain)=1;
% train.label(2,1:ntrain)=0;
% [train.trainx(1:ntrain),train.trainz(1:ntrain)]=ginput(ntrain);
% pause(1)
% figure(2)
% imagesc(image_input);
% 
% % Pick No-crack points
% title('Pick Non-Crack Points','fontsize',22);
% train.label(ntrain+1:2*ntrain)=0;
% train.label(2,ntrain+1:2*ntrain)=1;
% [train.trainx(ntrain+1:2*ntrain),train.trainz(ntrain+1:2*ntrain)]=ginput(ntrain);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            Step 2 Convert Image into Image patches             % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[nz,nx]=size(image_input);
image_pad=padimage(image_input,wdz,wdx);
%%% Obtain image batches from full image
image_slice=zeros(image_z,image_x,nz*nx);
image_slice_new=zeros(image_z_new,image_x_new,nz*nx);
[Xq,Zq]=meshgrid(1:(image_x-1)/(image_x_new-1):image_x,1:(image_z-1)/(image_z_new-1):image_z);
ii=1;
for ix=1:nx
    for iz=1:nz
        image_slice(1:image_z,1:image_x,ii)=image_pad(iz:iz+image_z-1,ix:ix+image_x-1);
        image_slice_new(:,:,ii)=interp2(image_slice(:,:,ii),Xq,Zq);
        ii=ii+1;
    end
end

%%% Obtain image batches from train set
train_slice=zeros(image_z,image_x,2*ntrain);
train_slice_new=zeros(image_z_new,image_x_new,2*ntrain);
[Xq,Zq]=meshgrid(1:(image_x-1)/(image_x_new-1):image_x,1:(image_z-1)/(image_z_new-1):image_z);
for in=1:2*ntrain
        train_slice(1:image_z,1:image_x,in)=...
            image_pad(train.trainz(in):train.trainz(in)+image_z-1,train.trainx(in):train.trainx(in)+image_x-1);
        train_slice_new(:,:,in)=interp2(train_slice(:,:,in),Xq,Zq);
end

clear image_slice train_slice

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  Step 3 Train CNN structure                    % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Input Training and Test set
train_x =train_slice_new;  % Training set
train_y =train.label;      % Training label
test_x =image_slice_new;   % Test set


rand('state',0)
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         Step 4 Input full image for Classification             % 
%   (Seperate The Image into 10 parts In case of out of memory)  %  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ii=1:9
disp(['Part ',num2str(ii),' Start']);
uu = cnnff(cnn, test_x(:,:,457*100*(ii-1)+1:457*100*ii));
[vv,ll]=max(uu.o);
ui=reshape(-ll+2,457,100);
un(:,100*(ii-1)+1:100*ii)=ui(:,1:100);

end
disp(['Part 10 Start']);
uu = cnnff(cnn, test_x(:,:,900*457+1:977*457));
[vv,ll2]=max(uu.o);
ui2=reshape(-ll2+2,457,77);
un(:,901:977)=ui2(:,1:77);
disp(['Done...']);


%%%%%%%%%% Image the result %%%%%%%%%%%%
figure(1)
subplot(1,2,1)
imagesc(image_input)
title({['Input Image']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Pixels','fontsize',22)
ylabel('Pixels','fontsize',22)
subplot(1,2,2)
imagesc(un)
title({['CNN Labels']},'fontsize',22,'color','black')
set(gca,'fontsize',22)
xlabel('Pixels','fontsize',22)
ylabel('Pixels','fontsize',22)
set(gcf,'unit','centimeters','position',[12 5 30 13])
colormap(parula)
% %%%%%%%%%Plot the error curve  %%%%%%%%%%%
% figure(2)
% plot(cnn.rL);
% title({['Misfit Curve']},'fontsize',22,'color','black')
% set(gca,'fontsize',22)
% xlabel('Iteration','fontsize',22)
% ylabel('Misfit','fontsize',22)



