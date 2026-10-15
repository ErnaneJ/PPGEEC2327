clear all,close all
% Setup training data
y=zeros(1,2000);
for ii=1:100
    y(1,20*ii)=2*rand(1)-1;
end

 rick=ricker(1000,1,0.1);
x_conv=conv(rick,y);
x(1,1:2000)=x_conv(1,11:2010);

%n=x;m=y;y=n;x=m;
% x(1,:)=1:400;
% y(1,:)=sind(1:400);
x = con2seq(x);
y = con2seq(y);

% Setup RNN Network
lrn_net = layrecnet(1,[10],'trainlm');%layrecnet(layerDelays,hiddenSizes,trainFcn) 
view(lrn_net)

% Setup RNN parameters
lrn_net.trainParam.epochs = 500; %Epoch number
lrn_net.trainParam.lr = 0.05; %Learning Rate
lrn_net.trainParam.min_grad = 1e-6; %Minimum gradient
lrn_net.trainParam.goal = 1e-15; % Loss function threshold

% Train RNN
lrn_net = train(lrn_net,x,y);

% Run RNN on train data
p = lrn_net(x);

%Plot train data
figure(2)
subplot(2,1,1)
plot(cell2mat(x))
lgd.FontSize=18;
title('Seismogram (Training data)','fontsize',20)
subplot(2,1,2)
plot(cell2mat(y))
hold on
plot(cell2mat(p))
lgd=legend('True','Predict');
lgd.FontSize=18;
title('Reflectivity (Training data)','fontsize',20)



% Setup Test data
test_y=zeros(1,200);
test_y(1,20)=0.5;
test_y(1,40)=0.7;
test_y(1,60)=-0.2;
test_y(1,80)=-0.3;
test_y(1,100)=0.7;
test_y(1,120)=0.9;
test_y(1,140)=-0.8;
test_y(1,160)=-0.3;
test_y(1,180)=0.5;
rick=ricker(1000,1,0.1);
test_x_conv=conv(rick,test_y);
test_x(1,1:200)=test_x_conv(1,11:210);
% Run RNN on test data
test_x = con2seq(test_x);
test_p = lrn_net(test_x);

%Plot test data
figure(3)
subplot(2,1,1)
plot(cell2mat(test_x))
lgd.FontSize=18;
title('Seismogram (Test data)','fontsize',20)
subplot(2,1,2)
plot(test_y)
hold on
plot(cell2mat(test_p))
lgd=legend('True','Predict');
lgd.FontSize=18;
title('Reflectivity (Test data)','fontsize',20)
