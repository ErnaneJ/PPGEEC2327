function Displaynn(M,res,nit,ww,layer_size,x,t,act_option)
tp=t*0;
layer_num=numel(layer_size); %layer number include the input and output

aa{1}=x;
% Do forward propagation
for iter =1:layer_num-1
    % ones(1, m) is for bias
    aa{iter} = [ones(1, M); aa{iter}];
    %%%----- z[n]=W[n]a[n-1]  ------%%%
    zz{iter}=ww{iter}*aa{iter};
    %%%----- a[n]=g(z[n])------%%%
    if iter ==layer_num-1 %output layer has to use sigmoid
        [aa{iter+1},~] = activation(zz{iter},1);
    else
        [aa{iter+1},~] = activation(zz{iter},act_option);
    end
end
tp=aa{layer_num}';

titleNN=[];
for iter =1:layer_num %%%----- N (in book) ------%%%
    titleNN=[titleNN,' ',num2str(layer_size(iter))];
end

subplot(121);plot(res(2:nit));
ylabel('Avg Misfit');xlabel('Iteration #')
title({[' NN convergence plot '];[ 'Total layer number =',num2str(layer_num)]});
subplot(122);plot([1:M],t,'r*',[1:M],tp,'g*')
ylabel('Example Class');xlabel('Example #')
title({[' NN result (Obs=red, Pred=green) ']; [ 'NN struture =',titleNN]});

