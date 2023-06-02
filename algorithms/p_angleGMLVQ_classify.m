function [estimatedLabels, varargout] = p_angleGMLVQ_classify(allX, actModel)
%angleGMLVQ_classify.m - classifies the given data with the given model
% TODO update! 
% example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=angleGMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = angleGMLVQ_classify(trainSet, GMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  model    : GMLVQ model with prototypes w their labels c_w, beta as parameter turning angles to distances and relevances A if applicable
% 
% output    : the estimated labels
% optional output:
% angles
% dists
%  
% Kerstin Bunte
% kerstin.bunte@googlemail.com
% Tue Apr 20 11:46 (GMT+1) 2016
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
if isfield(actModel,'theta')
    theta = actModel.theta;
elseif isfield(actModel,'beta')
    theta = actModel.beta;
    actModel.theta=actModel.beta;
else
    theta=[];
end
if isempty(actModel.theta)
     ca2d = @(cosa,beta) 0.5-0.5*cosa;
    
else
%     ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
    ca2P = @(cosa,theta) (exp( theta*cosa+theta)-1)/(exp(2*theta)-1);
end
if isfield(actModel,'A')
    allX(isnan(allX)) = 0;
    if iscell(actModel.A)
        if length(actModel.c_w)~=length(actModel.A), c_A = unique(round(actModel.c_w));classwise = 1;else c_A = actModel.c_w;classwise = 0;end
        xAw = nan(size(allX,1),length(actModel.c_w));
        normxA = nan(size(allX,1),length(actModel.c_w));
        normwA = nan(length(actModel.c_w),1);
        cosa = nan(size(allX,1),length(actModel.c_w));
        for j=1:length(actModel.c_w)
            if classwise, useAidx = find(c_A==actModel.c_w(j));else useAidx = j;end
            xAw(:,j) = allX*actModel.A{useAidx}'*actModel.A{useAidx}*actModel.w(j,:)';
            normwA(j) = sqrt(diag(actModel.w(j,:)*actModel.A{useAidx}'*actModel.A{useAidx}*actModel.w(j,:)'));
            normxA(:,j) = sqrt(diag(allX*actModel.A{useAidx}'*actModel.A{useAidx}*allX'));
            cosa(:,j) = bsxfun(@rdivide,xAw(:,j),normxA(:,j)*normwA(j)');
        end
    else
        if size(actModel.A,1)>1
            xAw = allX*actModel.A'*actModel.A*actModel.w';
            normxA = sqrt(sum((allX*actModel.A').^2,2));        % sqrt(diag(allX*actModel.A'*actModel.A*allX'));
            normwA = sqrt(sum((actModel.w*actModel.A').^2,2));  % sqrt(diag(actModel.w*actModel.A'*actModel.A*actModel.w'));
            cosa = bsxfun(@rdivide,xAw,normxA*normwA');            
%             (allX(1,:)*actModel.A')/sqrt(diag(allX(1,:)*actModel.A'*actModel.A*allX(1,:)'))
%             (actModel.A*actModel.w(1,:)')/sqrt(diag(actModel.w(1,:)*actModel.A'*actModel.A*actModel.w(1,:)'))            
        else
            cosa = bsxfun(@rdivide,bsxfun(@times,allX,actModel.A.^2)*actModel.w', sqrt(sum(bsxfun(@times,allX.^2,actModel.A.^2),2))*sqrt(sum(bsxfun(@times,actModel.w.^2,actModel.A.^2),2))');
        end
    end
%     dists = ca2d(cosa,beta);
    eD = ca2P(cosa,theta);    
%                 dists = squaredEuclidean(allX*actModel.A', actModel.w*actModel.A')
%                 eD = exp(-dists);
%                 p_x_given_w = eD./sum(eD,2);
%                 [~,idx] = max(p_x_given_w,[],2);
%                 [dists,p_x_given_w,idx,[trainLab;Y(testIdx)]]
%                 fprintf('Total error: %.3f\n',mean([trainLab;Y(testIdx)]~=idx));
else
    cosa = angle(allX,actModel.w);
    eD = ca2P(cosa,theta);
%     dists = ca2d(cosa,beta);
end
% [~,idx] = min(dists,[],2);
% estimatedLabels = actModel.c_w(idx);
% eD = exp(-dists); % this does not work for some reason
% eD = 1./dists;
probs = eD./sum(eD,2);

[~,idx] = max(probs,[],2);

estimatedLabels = actModel.c_w(idx);

nout = max(nargout,1)-1;
%%% additional output
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {probs};
        case(2)
			varargout(k) = {cosa};
	end
end
end
