clear all,close all
% Setup training data
x(1,:)=1:400;
y(1,:)=sind(1:400);
x = con2seq(x);
y = con2seq(y);

% Setup RNN Network
lrn_net = layrecnet(1,[10],'trainbr');%layrecnet(layerDelays,hiddenSizes,trainFcn) 
view(lrn_net)

% Setup RNN parameters
lrn_net.trainParam.epochs = 100; %Epoch number
lrn_net.trainParam.lr = 0.05; %Learning Rate
lrn_net.trainParam.min_grad = 1e-6; %Minimum gradient
lrn_net.trainParam.goal = 1e-10; % Loss function threshold

% Train RNN
lrn_net = train(lrn_net,x,y);

% Run RNN on train data
p = lrn_net(x);

%Plot train data
figure(2)
plot(cell2mat(y))
hold on
plot(cell2mat(p))
lgd=legend('True','Predict');
lgd.FontSize=18;
title('RNN on Train data','fontsize',20)



% Setup Test data
test_x=51:451;
test_y=sind(51:451);
test_x=con2seq(test_x);

% Run RNN on test data
test_p = lrn_net(test_x);

%Plot test data
figure(3)  
plot(test_y)
hold on
plot(cell2mat(test_p))
lgd=legend('True','Predict');
lgd.FontSize=18;
title('RNN on Test data','fontsize',20)
