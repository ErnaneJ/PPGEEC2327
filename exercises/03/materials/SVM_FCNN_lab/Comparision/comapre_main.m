clear
clc
close all

%%%%%%%%%%%%%%Comparision of ML methods in Identifyting migration artifacts
%%%%%%%%%%%%%%and reflector points%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                       Build the training sets                          %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% Loading migration image and features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load 'mig_and_features.mat'
[nz,nx]=size(mig);          % get the size of the model

%%%%The loaded data has four parts
%%%%mig    :: migration image with reflectors and artifacts
%%%%cohe   :: coherency feature
%%%%angl   :: local angle feature
%%%%energy :: Amplitude feature

%%%%%%%% Recale features by [x-mean(x)]/[max(x)-min(x)]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Rescale the coherency
cohe_vec=reshape(cohe,1,nz*nx);           % Reshape the 2D features to 1D
cohe_mean=mean(cohe_vec);                 % Compute the mean of the features
cohe_std=max(cohe_vec)-min(cohe_vec);     % Compute difference between the maximum and minimum value of the feature
cohe_vec2=(cohe_vec-cohe_mean)./cohe_std; % Feature scaling

%Rescale the angle (same as above)
angl_vec=reshape(angl,1,nz*nx);
angl_mean=mean(angl_vec);
angl_std=max(angl_vec)-min(angl_vec);
angl_vec2=(angl_vec-angl_mean)./angl_std;

%Rescale the energy (same as above)
engy_vec=reshape(engy,1,nz*nx);
engy_mean=mean(engy_vec);
engy_std=max(engy_vec)-min(engy_vec);
engy_vec2=(engy_vec-engy_mean)./engy_std;

% reshape the scaled 1D feature to 2D
cohe2=reshape(cohe_vec2,nz,nx);
angl2=reshape(angl_vec2,nz,nx);
engy2=reshape(engy_vec2,nz,nx);

% Plot the original image and it's feature map
figure(1);subplot(3,2,1);imagesc(cohe);colorbar;title('Coherency')
subplot(3,2,2);imagesc(cohe2);colorbar;title('Coherency after rescale')
subplot(3,2,3);imagesc(angl);colorbar;title('Local Angle')
subplot(3,2,4);imagesc(angl2);colorbar;title('Angle after rescale')
subplot(3,2,5);imagesc(engy);colorbar;title('Amplitude')
subplot(3,2,6);imagesc(engy2);colorbar;title('Amplitude after rescale')

%%%%%%% Build Training Sets
%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_refl=30;n_arti=50;       % set the number of reflector and artifact points as n_refl and n_arti, respectively. 
disp(['Please click ',num2str(n_arti),' artifacts points'])
pause(1.5)
figure(2);imagesc(Normalize(mig),[-0.06 0.06]);colormap(gray);
[x,z]=ginput(n_arti);      % here we use "ginput" function to pick the location of the artifact points. The picked results are saved in x and z, respectively
disp(['Please click ',num2str(n_refl),' reflector points'])
pause(1.5)
[x2,z2]=ginput(n_refl);    % here we use "ginput" function to pick the location of the reflector points. The picked results are saved in x2 and z2, respectively

n_total=n_refl+n_arti;     % compute the total number of training examples

X=zeros(n_total,3); % define the training dataset, which is (n_total by 3). 
                    % n_total : the number of training examples
                    % 3 : the number of features
for iz=1:n_refl     % build the training set for reflector points
    X(iz,1)=cohe2(round(z2(iz)),round(x2(iz)));
    X(iz,2)=angl2(round(z2(iz)),round(x2(iz)));
    X(iz,3)=engy2(round(z2(iz)),round(x2(iz)));
end 
kk=1;
for iz=n_refl+1:n_total   % build the training set for artifact points
    X(iz,1)=cohe2(round(z(kk)),round(x(kk)));
    X(iz,2)=angl2(round(z(kk)),round(x(kk)));
    X(iz,3)=engy2(round(z(kk)),round(x(kk)));
    kk=kk+1;
end
Y=zeros(n_total,1); Y(1:n_refl)=1;Y(n_refl+1:end)=0; % build the training set for outputs.
                                                     % Y(i) = 1 means this example is a reflector point
                                                     % Y(i) = 0 means this example is a artifact point

% 3D Plot of training dataset in the feature domain (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('(4) Display the training set')
figure(3)
for i=1:n_total
    if(Y(i)==0)
       plot3(X(i,1),X(i,2),X(i,3),'ko','MarkerFaceColor', 'y','MarkerSize', 7);
       hold on
    else
       plot3(X(i,1),X(i,2),X(i,3),'k+','LineWidth', 2,'MarkerSize', 7);
       hold on
    end
end
hold on
axis([min(cohe_vec2) max(cohe_vec2) min(angl_vec2) max(angl_vec2) min(engy_vec2) max(engy_vec2)]) % set the axis of the display
xlabel('Coherency'); ylabel('Angle'); zlabel('Amplitude')  % add label to the display
grid on; grid minor  % use dense grid in the display






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                           Logistic Regression                          %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%% Make decesion boundary by logistic regression %%%%%%%%%%%%
[m,n]=size(X);     % Get the size of training example
X2=[ones(m,1) X];  % Add intercept term to the training set

disp('Calculating decision boundarys')
maxiter=500;             % Iteration numbers
theta=rand(n+1,1)-0.5;   % Intialize the model parameters
step=10;                 % Set the step length

for iter=1:maxiter
    [cost, grad] = costFunction2(theta, X2, Y);          % Compute data misfit (cost) and gradient (grad)
    if(mod(iter,20)==0)                                  % Show the data misfit at every 20 iterations
      fprintf('iter= %d res = %f\n',iter,cost)
    end
    theta2=theta-step.*grad;                             % Update the model parameters with a trail step length
    [cost1, grad_temp] = costFunction2(theta2, X2, Y);   % Compute the new misfit (cost1) and gradient (grad_temp)
  %  fprintf('res1 = %f\n',cost1)
  
    if(cost1>cost)     % If the new misfit (cost1) is larger than the previous misfit (cost), then we reduce the step length and try again
       step=step*0.8;  % Reduce the step length
       theta2=theta-step.*grad;   % Update the model parameters with reduced trail step length
       [cost1, grad_temp] = costFunction2(theta2, X2, Y);  % Compute the new misfit
       fprintf('step = %f, res1 = %f\n',step,cost1)        % print the step length and data misfit
    end    % When the new misfit is smaller than the the misfit at the previous iteration (cost1 < cost), we update the model parameter
    
    theta=theta2; 
end

%%%%%%%%%%% 3D Plot of the decision boundary (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Display decision boundarys')
[X_surf1,Y_surf1]=meshgrid(min(cohe_vec2):0.01:max(cohe_vec2),min(angl_vec2):0.01:max(cohe_vec2));
Z_surf1=-(theta(1)+theta(2).*X_surf1+theta(3).*Y_surf1)./theta(4);  % compute the hyperplane predicted by the logistic regression
hold on
surf(X_surf1,Y_surf1,Z_surf1)  % display the hyperplane

%%%%% Use the decesion boundary to decide the all the imaging point 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Classify every image point in the migration image')

mig_result1=zeros(size(mig));  % mig_result1 has the same size as the migration image will save the predicted result
for ix=1:nx                               % Looping along x direction
    for iz=1:nz                           % Looping along z direction 
        val=[1 cohe2(iz,ix) angl2(iz,ix) engy2(iz,ix)];  % For each image point, get it's corresponding 3 features and add 1 bias term at the begining.  
%        val2=(val*[bb; ww]); 
        val2=1/(1+exp(-(val*theta)));     % Compute the predicted result.
        if(val2>0.5)                      % If the predicted result > 0.5, we set as 1. 
            val2=1;
        else                              % Or we set as 0.
            val2=0;
        end
        mig_result1(iz,ix)=val2;
    end
end
figure(4);imagesc(mig_result1)
title('Classified Results')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                            Neural Network                              %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%Nonliear Neural Wrok%%%%%%%%%%%%%%%%%%%%%%%%%
%In this section, we will create a 2 layer neural work (one hidden layer 
%and one output layer). Also, to increase the nonliearity, we increase the 
%freedom of input features from 1 to 2. (x1,x2,x3) --> (x1,x2,x3,x1^2,x2^2,x3^3,x1x2,x1x3,x2x3)
%If we increase the input features from 1 to 3, then we have 
%(x1,x2,x3) --> (x1,x2,x3,x1^2,x2^2,x3^2,x1^3,x2^3,x3^3,...)

%%%%%% Increase the freedom of the input features in the trainig set
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dimen=10;    
X2=zeros(m,dimen); 
X2(:,1)=1;                                 % Interception term
X2(:,2:4)=[X(:,1) X(:,2) X(:,3)];          % Original input features
X2(:,5:7)=[X(:,1).^2 X(:,2).^2 X(:,3).^2]; % The second power of the original features
X2(:,8)=X(:,1).*X(:,2);                    % Multiplication between different features  
X2(:,9)=X(:,1).*X(:,3);
X2(:,10)=X(:,2).*X(:,3);

%%%%%%% Set the number of unit in each layer
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[m1,n1]=size(X2);                 % Get the size of training set
input_layer_size = size(X2,2)-1;  % Set the number of node in the input layers
hidden_layer_size = 9;            % Set the number of node in the hidden layers
num_labels = 1;                   % Set the number of node in the output layers

%%%%%%% Initialize the model parameters in each layer by random
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

initial_theta1=(rand(input_layer_size+1,hidden_layer_size)-0.5)';
initial_theta2=(rand(hidden_layer_size+1,num_labels)-0.5)';

% Unroll parameters to vector
nn_params = [initial_theta1(:) ; initial_theta2(:)];

%%%%%%% Training (same workflow as logistic regression)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lamda=0.0; step=10; maxiter=500;  % lamda: regularization term (higher lamda can avoid over-fitting but will lost accuracy)
for iter=1:maxiter
    [cost,grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...  % nnCostFunction is the kernel used to compute the data misfit and gradient
                    num_labels, X2, Y, lamda); 
     if(mod(iter,50)==0||iter==1)                         
        fprintf('iter= %d, res = %f\n',iter,cost)
     end
     nn_params2=nn_params-step*grad; 
     [cost1,grad_temp] = nnCostFunction(nn_params2, input_layer_size, hidden_layer_size, ...
                    num_labels, X2, Y, lamda); 
     while(cost1>cost)
        step=step*0.8;
        nn_params2=nn_params-step*grad; 
        [cost1,grad_temp] = nnCostFunction(nn_params2, input_layer_size, hidden_layer_size, ...
                    num_labels, X2, Y, lamda);   
        fprintf('step = %f, res1 = %f\n',step,cost1)                      
     end
     nn_params=nn_params2;
end

%%%%%%%% Plot the traning set in the 3D feature domain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:70
    if(Y(i)==0)
       plot3(X(i,1),X(i,2),X(i,3),'ko','MarkerFaceColor', 'y','MarkerSize', 7);
       hold on
    else
       plot3(X(i,1),X(i,2),X(i,3),'k+','LineWidth', 2,'MarkerSize', 7);
       hold on
    end
end
hold on
axis([min(cohe_vec2) max(cohe_vec2) min(angl_vec2) max(angl_vec2) min(engy_vec2) max(engy_vec2)])
xlabel('Coherency'); ylabel('Angle'); zlabel('Amplitude')
grid on; grid minor

%%%%%%%%%%%%% 3D Plot of the decision boundary (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dx=0.01;
xx1=min(cohe_vec2):dx:max(cohe_vec2);
xx2=min(angl_vec2):dx:max(angl_vec2);
xx3=min(engy_vec2):dx:max(engy_vec2);

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

vals=visualizeBoundaryNN3D([xx1' xx2' xx3'], Theta1, Theta2,dx);
[X_surf2,Y_surf2]=meshgrid(min(cohe_vec2):dx:max(cohe_vec2),min(angl_vec2):dx:max(angl_vec2));
[nz2,nx2,ny2]=size(vals);
vals2=zeros(size(vals));
for iy=1:ny2
   for iz=1:nz2 
       vals2(iz,1:nx2-1,iy)=diff(vals(iz,:,iy));
   end
end
Z_surf2=-2*ones(size(X_surf2));
z_range=min(engy_vec2):dx:max(engy_vec2);
trace=zeros(1,ny2);
for iz=nz2:-1:1
for ix=nx2-1:-1:1
    trace(:)=vals2(iz,ix,:);
    [val,loc]=max(trace);
 if(loc>1)
       Z_surf2(iz,ix)=z_range(loc);
    else
       Z_surf2(iz,ix)=Z_surf2(iz,ix+1); 
  end
end
end
hold on
surf(X_surf2,Y_surf2,Z_surf2)


%%%%%% Prediction. The trained model parameters are saved in Theta1 and Theta2. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mig_result2=zeros(size(mig));
for ix=1:nx
    for iz=1:nz
        val=[1 cohe2(iz,ix) angl2(iz,ix) engy2(iz,ix) ...   % For each image point, get it's corresponding features
             cohe2(iz,ix)^2 angl2(iz,ix)^2 engy2(iz,ix)^2 ...
             cohe2(iz,ix)*angl2(iz,ix) cohe2(iz,ix)*engy2(iz,ix) ...
             angl2(iz,ix)*engy2(iz,ix)];
         
        h1 = sigmoid(val * Theta1');  
        h2 = sigmoid([ones(size(h1,1), 1) h1] * Theta2');   % Predice the values
        
        if(h2>0.5)     % If the predicted result > 0.5, we set as 1. Otherwise we set as 0
            h2=1;
        else
            h2=0;
        end
        mig_result2(iz,ix)=h2;
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------------------------%
%                        Support Vector Machine                          %
%------------------------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% For your homework you will need to re-build the training set X and Y. 
%%%% Do it here... 



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Support Vector Machine%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Make decesion boundary by SVM
C = 1;              % The C parameter is a positive value that controls the
                    % penalty for misclassified training examples. A large 
                    % C parameter tells the SVM to try to classify all the 
                    % examples correctly. C plays a role similar to
                    % 1/lamda, where lamda is the regularization parameter                   
sigma = 0.1;        % sigma controls the shape of gaussian function. In other
                    % words, how fast the similarity metric decrease

% "svmTrain" is the matlab function used for SVM training. Here we choose
% Gaussian as the kernel. If you want to use the linear kernel, please use
% model= svmTrain(X, Y, C, @linearKernel); 
model= svmTrain(X, Y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));  
                                                                    
                                                                  
%%%%%%%%%%% 3D plot the training set (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:70
    if(Y(i)==0)
       plot3(X(i,1),X(i,2),X(i,3),'ko','MarkerFaceColor', 'y','MarkerSize', 7);
       hold on
    else
       plot3(X(i,1),X(i,2),X(i,3),'k+','LineWidth', 2,'MarkerSize', 7);
       hold on
    end
end
hold on
axis([min(cohe_vec2) max(cohe_vec2) min(angl_vec2) max(angl_vec2) min(engy_vec2) max(engy_vec2)])
xlabel('Coherency'); ylabel('Angle'); zlabel('Amplitude')
grid on; grid minor
xx1=min(cohe_vec2):0.1:max(cohe_vec2);
xx2=min(angl_vec2):0.1:max(angl_vec2);
xx3=min(engy_vec2):0.1:max(engy_vec2);

%%%%%%%%%%%%% 3D plot of the decision boundary (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vals=visualizeBoundary3D([xx1' xx2' xx3'], model);
[X_surf3,Y_surf3]=meshgrid(min(cohe_vec2):0.01:max(cohe_vec2),min(angl_vec2):0.01:max(angl_vec2));
[nz2,nx2,ny2]=size(vals);
vals2=zeros(size(vals));
for iy=1:ny2
   for iz=1:nz2 
       vals2(iz,1:nx2-1,iy)=diff(vals(iz,:,iy));
   end
end
Z_surf3=-2*ones(size(X_surf3));
z_range=min(engy_vec2):0.01:max(engy_vec2);
trace=zeros(1,ny2);
for iz=nz2:-1:1
for ix=nx2-1:-1:1
    trace(:)=vals2(iz,ix,:);
    [val,loc]=max(trace);
 if(loc>1)
       Z_surf3(iz,ix)=z_range(loc);
    else
       Z_surf3(iz,ix)=Z_surf3(iz,ix+1); 
  end
end
end

hold on
surf(X_surf3,Y_surf3,Z_surf3)

%%%%%%%%%%% Prediction. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bb=model.b;  % model.b is the trained model parameters
ww=model.w;  % model.w is the trained bias term
mig_result3=zeros(size(mig));
for ix=1:nx
    for iz=1:nz
        val=[cohe2(iz,ix) angl2(iz,ix) engy2(iz,ix)]; % For each image point, get it's corresponding 3 features
        val2 = svmPredict2(model, val);  % "svmPredict2" is used for prediction.
        mig_result3(iz,ix)=val2;
    end
end

%%%%%%%%%%%% Plot all (Please do not change this part)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure;subplot(1,3,1);imagesc(mig_result1);title('Logistic Regression')
subplot(1,3,2);imagesc(mig_result2);title('Neural Network')
subplot(1,3,3);imagesc(mig_result3);title('SVM')

figure;subplot(1,3,1);
for i=1:70
    if(Y(i)==0)
       plot3(X(i,1),X(i,2),X(i,3),'ko','MarkerFaceColor', 'y','MarkerSize', 7);
       hold on
    else
       plot3(X(i,1),X(i,2),X(i,3),'k+','LineWidth', 2,'MarkerSize', 7);
       hold on
    end
end
hold on
axis([min(cohe_vec2) max(cohe_vec2) min(angl_vec2) max(angl_vec2) min(engy_vec2) max(engy_vec2)])
xlabel('Coherency'); ylabel('Angle'); zlabel('Amplitude')
grid on; grid minor
hold on
surf(X_surf1,Y_surf1,Z_surf1);title('Logistic Regression')

subplot(1,3,2)
for i=1:70
    if(Y(i)==0)
       plot3(X(i,1),X(i,2),X(i,3),'ko','MarkerFaceColor', 'y','MarkerSize', 7);
       hold on
    else
       plot3(X(i,1),X(i,2),X(i,3),'k+','LineWidth', 2,'MarkerSize', 7);
       hold on
    end
end
hold on
axis([min(cohe_vec2) max(cohe_vec2) min(angl_vec2) max(angl_vec2) min(engy_vec2) max(engy_vec2)])
xlabel('Coherency'); ylabel('Angle'); zlabel('Amplitude')
grid on; grid minor
hold on
surf(X_surf2,Y_surf2,Z_surf2);title('Neural Network')

subplot(1,3,3)
for i=1:70
    if(Y(i)==0)
       plot3(X(i,1),X(i,2),X(i,3),'ko','MarkerFaceColor', 'y','MarkerSize', 7);
       hold on
    else
       plot3(X(i,1),X(i,2),X(i,3),'k+','LineWidth', 2,'MarkerSize', 7);
       hold on
    end
end
hold on
axis([min(cohe_vec2) max(cohe_vec2) min(angl_vec2) max(angl_vec2) min(engy_vec2) max(engy_vec2)])
xlabel('Coherency'); ylabel('Angle'); zlabel('Amplitude')
grid on; grid minor
hold on
surf(X_surf3,Y_surf3,Z_surf3);title('SVM')



