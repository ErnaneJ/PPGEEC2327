function vals=visualizeBoundaryNN3D(X, theta1, theta2,dx)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
%plotData(X, y)
[X1,X2]=meshgrid(min(X(:,1)):dx:max(X(:,1)),...
                min(X(:,2)):dx:max(X(:,2)));
X3=min(X(:,3)):dx:max(X(:,3));
[nz,nx]=size(X1);
vals = zeros(nz,nx,length(X3));


for j=1:length(X3)
  temp=X3(j)*ones(size(X1(:,1)));
  for i = 1:size(X1, 2)
      this_X = [X1(:, i), X2(:, i),temp];
      XX_temp=zeros(size(this_X,1),10);
      XX_temp(:,1)=1;
      XX_temp(:,2:4)=[this_X(:,1) this_X(:,2) this_X(:,3)];
      XX_temp(:,5:7)=[this_X(:,1).^2 this_X(:,2).^2 this_X(:,3).^2];
      XX_temp(:,8)=this_X(:,1).*this_X(:,2);
      XX_temp(:,9)=this_X(:,1).*this_X(:,3);
      XX_temp(:,10)=this_X(:,2).*this_X(:,3);
      h1 = sigmoid(XX_temp * theta1');
      h2 = sigmoid([ones(size(h1,1), 1) h1] * theta2');
       
      for k=1:size(h2,1)
        if(h2(k)>=0.5)
           vals(k,i,j) = 1;
        else
           vals(k,i,j) = 0; 
        end
      end
  end
end

% Plot the SVM boundary
% hold on
% contour(X1, X2, vals, [0.5 0.5], 'b');
% hold off;

end
