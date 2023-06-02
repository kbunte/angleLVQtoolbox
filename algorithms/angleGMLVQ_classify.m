function [estimatedLabels, varargout] = angleGMLVQ_classify(allX, actModel)
%angleGMLVQ_classify.m - classifies the given data with the given model
%  example for usage:
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
if isfield(actModel,'beta')
    beta = actModel.beta;
else
    beta=[];
end
if isempty(actModel.beta)
    ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
end
if isfield(actModel,'A')
    allX(isnan(allX)) = 0;
    if iscell(actModel.A)
        if length(actModel.c_w)~=length(actModel.A) 
            c_A = unique(actModel.c_w); % redundant! can be worked without! % c_A = unique(round(actModel.c_w));
            classwise = 1;
        else
            c_A = actModel.c_w;
            classwise = 0;
        end
        xAw = nan(size(allX,1),length(actModel.c_w));
        normxA = nan(size(allX,1),length(actModel.c_w));
        normwA = nan(length(actModel.c_w),1);
        cosa = nan(size(allX,1),length(actModel.c_w));
        if size(actModel.A{1},1)>1
            for j=1:length(actModel.c_w)
                if classwise, useAidx = find(c_A==actModel.c_w(j));else useAidx = j;end
                xAw(:,j) = allX*actModel.A{useAidx}'*actModel.A{useAidx}*actModel.w(j,:)';
                normwA(j) = sqrt(diag(actModel.w(j,:)*actModel.A{useAidx}'*actModel.A{useAidx}*actModel.w(j,:)'));
                normxA(:,j) = sqrt(diag(allX*actModel.A{useAidx}'*actModel.A{useAidx}*allX'));
                cosa(:,j) = bsxfun(@rdivide,xAw(:,j),normxA(:,j)*normwA(j)');
            end
        else
            for j=1:length(actModel.c_w)
                if classwise, useAidx = find(c_A==actModel.c_w(j));else useAidx = j;end
                R=diag(actModel.A{useAidx});
                xAw(:,j) = allX*R'*R*actModel.w(j,:)'; 
                normwA(j) =  sqrt(diag(actModel.w(j,:)*R'*R*actModel.w(j,:)'));
                normxA(:,j)= sqrt(diag(allX*R'*R*allX'))';   % 
                cosa(:,j) = bsxfun(@rdivide,xAw(:,j),normxA(:,j)*normwA(j)');
            end
        end
    else
        if size(actModel.A,1)>1
            xAw = allX*actModel.A'*actModel.A*actModel.w';
            normxA = sqrt(diag(allX*actModel.A'*actModel.A*allX'));
            normwA = sqrt(diag(actModel.w*actModel.A'*actModel.A*actModel.w'));
            cosa = bsxfun(@rdivide,xAw,normxA*normwA');
        else
            cosa = bsxfun(@rdivide,bsxfun(@times,allX,actModel.A.^2)*actModel.w', sqrt(sum(bsxfun(@times,allX.^2,actModel.A.^2),2))*sqrt(sum(bsxfun(@times,actModel.w.^2,actModel.A.^2),2))');
        end
    end
    dists = ca2d(cosa,beta);
else
    cosa = angle(allX,actModel.w);
    dists = ca2d(cosa,beta);
end
[~,idx] = min(dists,[],2);
estimatedLabels = actModel.c_w(idx);

nout = max(nargout,1)-1;
%%% additional output
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {cosa};
        case(2)
			varargout(k) = {dists};
	end
end
end
