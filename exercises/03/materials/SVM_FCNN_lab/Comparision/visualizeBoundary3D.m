function vals=visualizeBoundary3D(X, model)
%VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
%   boundary learned by the SVM and overlays the data on it

% Plot the training data on top of the boundary
%plotData(X, y)
[X1,X2]=meshgrid(min(X(:,1)):0.01:max(X(:,1)),...
                min(X(:,2)):0.01:max(X(:,2)));
X3=min(X(:,3)):0.01:max(X(:,3));
[nz,nx]=size(X1);
vals = zeros(nz,nx,length(X3));

for j=1:length(X3)
  temp=X3(j)*ones(size(X1(:,1)));
  for i = 1:size(X1, 2)
      this_X = [X1(:, i), X2(:, i),temp];
      vals(:, i,j) = svmPredict2(model, this_X);
  end 
end
    
end
