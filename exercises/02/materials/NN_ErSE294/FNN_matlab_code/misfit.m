function [ J,in ] = misfit( Y_pred,Y_obs,scale,type)
%misfit function
%
%input:
%Y_pred: predicted value
%Y_obs: observed vakue
%scale: scale the misift
%type: objetive function type: 1 for L2, 2 for likelihood
%
%e.g: J=0.5*scale*||Y_pred-Y_obs)||2
%output:
% J is the value of the objective function
% in is the gradient of J with respective to Y_pred

if type==1 % for L2 norm objective function
    %J = sum( sum( (Y_pred-Y_obs).^2 ) );
    J = sqrt(sum( sum( (Y_pred-Y_obs).^2 ) ));  %adjust from Jerry
    in = 2*(Y_pred-Y_obs);
elseif type==2 % for likelihood  objective function
    J = sum(sum(-Y_obs.*log(Y_pred) - (1-Y_obs).*log(1-Y_pred)));
    in = -Y_obs./Y_pred + (1-Y_obs)./(1-Y_pred);  % dJ / d Y_pred
else
    display('You entered the wrong type for the misfit function');
    stop
end

J=J*scale;in=in*scale;

end


