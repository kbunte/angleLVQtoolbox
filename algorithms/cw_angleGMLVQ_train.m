function [model, varargout] = cw_angleGMLVQ_train(trainSet, trainLab, varargin)
%angleGMLVQ_train.m - trains the angle Generalized Matrix LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=cw_angleGMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = angleGMLVQ_classify(trainSet, GMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  trainLab : vector with the labels of the training set
% optional parameters:
%  costWeight        : (default=ones(nb of classes)) weights for classification errors following the confusion matrix with true class in rows and predicted in columns 
%  PrototypesPerClass: (default=1) the number of prototypes per class used. This could be a number or a vector with the number for each class
%  beta              : (default=1) the beta parameter for turning cos a into distance
%  dim               : (default=nb of features for training) the maximum rank or projection dimension. 1 learns a relevance vector instead of a matrix
% if dim is a vector of length k either with respect to the number of
% classes C or the number of prototypes k
%  regularization    : (default=0) values usually between 0 and 1 treat with care. Regularizes the eigenvalue spectrum of A'*A to be more homogeneous
%  initialPrototypes : (default=[]) a set of prototypes to start with. If not given initialization near the class means
%  initialRelevance  : (default=[]) relevances A to start with. If not given random initialization. If a vector do relevance vector learning
%  testSet           : (default=[]) an optional test set used to compute the test error. The last column is expected to be a label vector
%  comparable        : (default=0) a flag which resets the random generator to produce comparable results if set to 1
% parameter for the build-in function fminlbfgs
%  goalsExactAchieve: (default=1) 
%  Display          : (default=iter) the optimization output 'iter' or 'off'
%  GradObj          : (default=on) use the gradient information or not
%  HessUpdate       : (default=lbfgs) the update can be 'lbfgs', 'bfgs' or 'steepdesc'
%  GradConstr       : (default=0)
%  TolFun           : (default=1e-6) the tolerance
%  MaxIter          : (default=2500) the maximal number of iterations
%  MaxFunEvals      : (default=1000000) the maximal number of function evaluations
%  TolX             : (default=1e-10) tolerance
%  DiffMinChange    : (default=1e-10) minimal change
%
% output: the GMLVQ model with prototypes w their labels c_w and the matrix omega 
%  optional output:
%  initialization : a struct containing the settings
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
% 
% Citation information:
% Kerstin Bunte, etc.:

nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainSet,1) & isnumeric(x));
p.addParameter('costWeight',[], @(x)(bsxfun(@isequal,size(x),ones(1,2).*length(unique(trainLab))) && isfloat(x)));
% p.addParameter('costWeights', @(x) ones(length(unique(trainLab))));
% p.addParameter('beta',1, @(x)(isfloat(x) && x>=0));
p.addParameter('beta',1, @(x)(isempty(x) || (isfloat(x) && x>=0)));
p.addParameter('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParameter('initialPrototypes',[], @(x)(size(x,2)-1==size(trainSet,2) && isfloat(x)));
p.addParameter('initialRelevance',[], @(x)(size(x,2)==size(trainSet,2) && isfloat(x)));
p.addParameter('dim',size(trainSet,2), @(x)( (length(x)>1)||(~(x-floor(x)) && x<=size(trainSet,2) && x>=0 ) ));
p.addParameter('regularization',0, @(x)(isfloat(x) && x>=0));
p.addParameter('testSet', [], @(x)(size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addParameter('comparable', 0, @(x)(~(x-floor(x))));
p.addParameter('optreps',1,@(x)(~(x-floor(x))));
% parameter for the build-in function
p.addParameter('GradConstr',0,@(x) isfloat(x));
p.addParameter('GoalsExactAchieve',1,@(x)(~(x-floor(x))));
p.addParameter('Display', 'iter', @(x)any(strcmpi(x,{'iter','off'})));
p.addParameter('GradObj', 'on', @(x)any(strcmpi(x,{'on','off'})));
p.addParameter('HessUpdate', 'lbfgs', @(x)any(strcmpi(x,{'lbfgs', 'bfgs', 'steepdesc'})));
p.addParameter('TolFun',1e-6,@(x) isfloat(x));
p.addParameter('MaxIter', 2500, @(x)(~(x-floor(x))));
p.addParameter('MaxFunEvals', 1000000, @(x)(~(x-floor(x))));
p.addParameter('TolX',1e-10,@(x) isfloat(x));
p.addParameter('DiffMinChange',1e-10,@(x)isfloat(x));
p.CaseSensitive = true;
p.FunctionName = 'cw_angleGMLVQ_train';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% optimization options
%   'OutputFcn','LVQ_progresser', ...
options = struct( ...
  'Display',p.Results.Display, ...
  'GradObj',p.Results.GradObj, ...
  'GradConstr',p.Results.GradConstr, ...
  'GoalsExactAchieve',p.Results.GoalsExactAchieve, ...
  'TolFun',p.Results.TolFun, ...
  'MaxIter',p.Results.MaxIter, ...
  'MaxFunEvals', p.Results.MaxFunEvals, ...
  'TolX',p.Results.TolX, ...
  'DiffMinChange',p.Results.DiffMinChange, ...
  'HessUpdate',p.Results.HessUpdate ...
);
%%% check if results should be comparable
if p.Results.comparable,rng('default');end

costWeight = p.Results.costWeight;%./sum(sum(p.Results.costWeights));
if ~isempty(costWeight) costWeight = costWeight/max(costWeight(:));end
%%% set useful variables
nb_samples = size(trainSet,1);
nb_features = size(trainSet,2);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

allClass = unique(trainLab);
nb_classes = length(allClass);

dim = p.Results.dim;
% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the eigenvalue spectrum of A''*A with ',num2str(regularization)]);end
if ~strcmpi(options.Display,'off')
% Display all arguments.
    disp 'Settings for angle GMLVQ:'
    disp(p.Results);
end

%%% check the number of prototypes per class if one integer is given and turn
%%% it into a vector
nb_ppc = p.Results.PrototypesPerClass;
if length(nb_ppc)~=nb_classes,nb_ppc = ones(1,nb_classes)*nb_ppc;end
if sum(nb_ppc)~=nb_classes, error('this is not implemented for more than 1 prototype per class yet!!!'); end

try
    nanmean(0);
    usenanmean = @(Mat) nanmean(Mat);
catch %ME
%     if (strcmp(ME.identifier,'MATLAB:UndefinedFunction'))
%       msg = ME.message;
%       causeException = MException('MATLAB:p_angleGMLVQ_train',msg);
%       ME = addCause(ME,causeException);
%     end
%     rethrow(ME)
    warning('Function nanmean unknown. Use own implementation instead!');
    usenanmean = @(Mat) arrayfun(@(j) mean( Mat(~isnan(Mat(:,j)),j) ),1:size(Mat,2));
end

%%% initialize the prototypes
if isempty(p.Results.initialPrototypes)
    % initialize near the class centers
    variables = cat(2,cell2mat(arrayfun(@(x) bsxfun(@times,allClass(x),ones(nb_ppc(x),1)),1:length(nb_ppc),'uni',0)'),nan(sum(nb_ppc),nb_features));
    w = cell2mat(arrayfun(@(x) bsxfun(@times,usenanmean(trainSet(trainLab==allClass(x),:)),ones(nb_ppc(x),nb_features)),1:length(nb_ppc),'uni',0)');
    w(isnan(w)) = 0;
    variables(:,2:end) = w;clear w;
    variables(:,2:end) = variables(:,2:end) +(rand(size(variables,1),nb_features)*2-ones(size(variables,1),nb_features))/10;
else
    % initialize with given prototypes [c_w,w]
    if ~isempty(setdiff(p.Results.initialPrototypes(:,1),unique(trainLab)))
        error('First column of initialPrototypes must contain all the labels in the training set!');
    end
    variables = p.Results.initialPrototypes;
end

%%% initialize the matrix
if isempty(p.Results.initialRelevance)
    if dim
        if(dim==1)
            variables = [variables;[nan,sqrt(ones(1,nb_features)./nb_features)]];
        else % initialize with random numbers between -1 and 1
%             variables = [variables;[nan,ones(1,nb_features)./nb_features]];
            if length(dim)>1
%                 mats = cell2mat(arrayfun(@(x) [nan(dim(x),1),rand(dim(x),nb_features)*2-ones(dim(x),nb_features)],1:length(dim),'uni',0)');
                mats = arrayfun(@(x) [rand(dim(x),nb_features)*2-ones(dim(x),nb_features)],1:length(dim),'uni',0);
                mats = cell2mat(cellfun(@(x) x/sqrt(trace(x'*x)),mats,'UniformOutput',0)');
                variables = [variables;nan(sum(dim),1),mats];
            else
                variables = [variables;[nan(dim,1),rand(dim,nb_features)*2-ones(dim,nb_features)]];
%disp('TODO normalize trace')
            end
        end
    end
else
    variables = [variables;[nan(size(p.Results.initialRelevance,1),1),p.Results.initialRelevance]];
end
beta = p.Results.beta;

LabelEqualsPrototype = bsxfun(@eq,trainLab,variables(~isnan(variables(:,1)),1)');
if length(dim)>1
%      error('not implemented!');
        useFun = @(a,b,c,d,e,f,g,h,i) cw_angleLGMLVQ_optfun(a,b,c,d,e,f,g,h,i);
else
    if dim==1
        useFun = @(a,b,c,d,e,f,g,h,i) cw_angleGRLVQ_optfun(a,b,c,d,e,f,g,h,i); % ,j
    else
       % error('not implemented!');
        useFun = @(a,b,c,d,e,f,g,h,i) cw_angleGMLVQ_optfun(a,b,c,d,e,f,g,h,i); % ,j
    end
end
% learn prototypes without relevances
% options.GradObj='off'
% save('test');error('load test')
[variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,trainSet,LabelEqualsPrototype,beta,regularization,1,1,dim,costWeight),variables,options); % ,trainLab
% [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,trainSet,LabelEqualsPrototype,beta,regularization,1,0,dim,costWeight,trainLab),variables,options);
% learn relevances without the prototypes
%[variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,trainSet,LabelEqualsPrototype,beta,regularization,0,1,dim,costWeight,trainLab),variables,options);
for actrep = 1:p.Results.optreps
    % learn prototypes without relevances
    [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,trainSet,LabelEqualsPrototype,beta,regularization,1,0,dim,costWeight),variables,options); % ,trainLab
    % learn relevances without the prototypes
    [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,trainSet,LabelEqualsPrototype,beta,regularization,0,1,dim,costWeight),variables,options); % ,trainLab
end
relIdx = find(isnan(variables(:,1)));
wIdx = 1:size(variables,1);wIdx(relIdx) = [];
model.c_w = variables(wIdx,1);
model.w = variables(wIdx,2:end);
model.beta = beta;
model.costs=costWeight;
if ~isempty(relIdx)
    if length(dim)>1
        Aidx = mat2cell(1:length(relIdx),1,dim);
        model.A = cellfun(@(x) variables(relIdx(x),2:end),Aidx,'uni',0);
        model.A = cellfun(@(x) x/sqrt(trace(x'*x)),model.A,'UniformOutput',0);
    else
        model.A = variables(relIdx,2:end);
        model.A = model.A/sqrt(trace(model.A'*model.A));
    end
end
%%% output of the training
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {fval};
		case(2)
			varargout(k) = {trainError};
		case(3)
            testSet = p.Results.testSet;
			varargout(k) = {testError};
	end
end
end

