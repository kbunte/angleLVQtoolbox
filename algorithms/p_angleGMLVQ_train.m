function [model, varargout] = p_angleGMLVQ_train(trainX, trainLab, varargin)
%angleGMLVQ_train.m - trains the angle Generalized Matrix LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=angleGMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = angleGMLVQ_classify(trainSet, GMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  trainLab : vector with the labels of the training set
% optional parameters:
% marginVersion      : (default=1) switch to change the costfunction, per default the margin version KL(q|p) is used, with 0 KL(p|q) is used
% costWeight         : (default=[]) user can specifiy which errors are more important to get right. Takes the form of costs in the confusion matrix with dimension equal to the number of classes, true class/predicted
%  theta             : (default=1) the theta parameter for turning cos a into softmax. For theta=[] linear conversion is used otherwise  exponential with slope theta
%  dim               : (default=nb of features for training) the maximum rank or projection dimension>1. The relevance version is not implemented
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
% output: the p_angleGMLVQ_margin model with prototypes w their labels c_w and the matrix omega 
%  optional output:
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
%  initialization : a struct containing the settings
% 
% Citation information:
% Kerstin Bunte, etc.:

nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainX,1) & isnumeric(x));
p.addParameter('marginVersion',1,@(x) isnumeric(x)); % tell if the margin version should be used for optimization, default is yes =1
p.addParameter('theta',1, @(x)(isempty(x) || (isfloat(x) && x>=0)));
p.addParameter('costWeight',[], @(x)(bsxfun(@isequal,size(x),ones(1,2).*length(unique(trainLab))) && isfloat(x)));
% p.addParameter('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParameter('initialPrototypes',[], @(x)(size(x,2)==size(trainX,2) && isfloat(x)));
p.addParameter('initialRelevance',[], @(x)(size(x,2)==size(trainX,2) && isfloat(x)));
p.addParameter('dim',size(trainX,2), @(x)( (length(x)>1)||(~(x-floor(x)) && x<=size(trainX,2) && x>=0 ) ));
p.addParameter('regularization',0, @(x)(isfloat(x) && x>=0));
p.addParameter('testSet', [], @(x)(size(x,2)-1)==size(trainX,2) & isfloat(x));
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
p.FunctionName = 'p_angleGMLVQ_train';
% Parse and validate all input arguments.
p.parse(trainX, trainLab, varargin{:});

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
% options = struct( 'Display','iter', 'GradObj','on', 'GradConstr',0, 'GoalsExactAchieve',1, 'TolFun',1e-6, 'MaxIter',2500, 'MaxFunEvals',1000000,'TolX',1e-10, 'DiffMinChange',1e-10, 'HessUpdate','lbfgs');
%%% check if results should be comparable
if p.Results.comparable,rng('default');end
%%% set useful variables
nb_samples = size(trainX,1);
nb_features = size(trainX,2);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

allClass = unique(trainLab);
nb_classes = length(allClass);

dim = p.Results.dim;
% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the eigenvalue spectrum of A''*A with ',num2str(regularization)]);end

% Display all arguments.
if ~strcmp(p.Results.Display,'off')
    disp 'Settings for probabilistic angle GMLVQ:';
    disp(p.Results); 
end

%%% check the number of prototypes per class if one integer is given and turn
%%% it into a vector
% nb_ppc = p.Results.PrototypesPerClass;
% if length(nb_ppc)~=nb_classes
nb_ppc = ones(1,nb_classes); % prob version only implemented for 1 prototype per class, more is requiring a different model
% end
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
    % compute class-wise centers
    w = cell2mat(arrayfun(@(x) bsxfun(@times,usenanmean(trainX(trainLab==allClass(x),:)),ones(nb_ppc(x),nb_features)),1:length(nb_ppc),'uni',0)');
    if ~isempty(find(isnan(w), 1))
        warning('classwise means show NaN, which only happens when a feature is completely missing for a class! This can have unintended consequences!');
        w(isnan(w)) = 0;
    end
    variables(:,2:end) = w;clear w;
    variables(:,2:end) = variables(:,2:end) +(rand(size(variables,1),nb_features)*2-ones(size(variables,1),nb_features))/10; % add slight rng deviations to avoid equivalent prototype starts
    % labels of the prototypes and nan to mark rows for omega 
    c_vars = [cell2mat(arrayfun(@(x) bsxfun(@times,allClass(x),ones(nb_ppc(x),1)),1:length(nb_ppc),'uni',0)');nan(dim,1)];
else
    % initialize with given prototypes [c_w,w]
    if ~isempty(setdiff(p.Results.initialPrototypes(:,1),unique(trainLab)))
        error('First column of initialPrototypes must contain all the labels in the training set!');
    end
    variables = p.Results.initialPrototypes;
    c_vars = variables(:,1);
end

%%% initialize the matrix
if isempty(p.Results.initialRelevance)
    if  ((dim==1) && length(dim)==1)
        variables = [variables;[nan,sqrt(ones(1,nb_features)./nb_features)]];
    else % initialize with random numbers between -1 and 1
        variables = [variables;[nan(dim,1),rand(dim,nb_features)*2-ones(dim,nb_features)]];
    end
else
    variables = [variables;[nan(size(p.Results.initialRelevance,1),1),p.Results.initialRelevance]];
end
% A = (rand(dim,nb_features)*2-ones(dim,nb_features))/10;
model.theta = p.Results.theta;
model.c_w = c_vars(~isnan(c_vars));
model.w = variables(1:length(model.c_w),2:end);
model.A = variables(length(model.c_w)+1:end,2:end); 
model.A=model.A/sqrt(trace(model.A'*model.A));
variables(length(model.c_w)+1:end,2:end) = model.A;

costWeight = p.Results.costWeight;
if ~isempty(costWeight) costWeight = costWeight/max(costWeight(:));end

LabelEqualsPrototype = bsxfun(@eq,trainLab,c_vars(~isnan(c_vars))');% model2=model;
% variables_old = variables
% estimatedLabels = p_angleGMLVQ_classify(trainX, model);trainError = mean( trainLab ~= estimatedLabels )
if isempty(costWeight)
    if p.Results.marginVersion
        disp('run D(q|p) version');
        useFun = @(a,b,c,d,e,f,g,h,i) p_angleGMLVQ_margin_optfun(a,b,c,d,e,f,g,h,i);
    else
        disp('run D(p|q) version');
        useFun = @(a,b,c,d,e,f,g,h,i) p_angleGMLVQ_optfun2(a,b,c,d,e,f,g,h,i);
    end
    [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,1,1,dim),variables,options);
% learn prototypes without the relevances
%     [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,1,0,dim),variables,options);
% learn relevances without the prototypes
%     [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,0,1,dim),variables,options);
    for actrep = 1:p.Results.optreps
    % learn prototypes without relevances
        [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,1,0,dim),variables,options);
    % learn relevances without the prototypes
        [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,0,1,dim),variables,options);
    end    
else
    if p.Results.marginVersion
        disp('run D(q|p) version');
        useFun = @(a,b,c,d,e,f,g,h,i,j) p_cw_angleGMLVQ_margin_optfun(a,b,c,d,e,f,g,h,i,j);
    else
        disp('run D(p|q) version');
        useFun = @(a,b,c,d,e,f,g,h,i,j) p_cw_angleGMLVQ_optfun(a,b,c,d,e,f,g,h,i,j);
    end
    [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,1,1,dim,costWeight),variables,options);
% learn prototypes without the relevances
%     [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,1,0,dim,costWeight),variables,options);
% learn relevances without the prototypes
%     [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,0,1,dim,costWeight),variables,options);
    for actrep = 1:p.Results.optreps
    % learn prototypes without relevances
        [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,1,0,dim, costWeight),variables,options);
    % learn relevances without the prototypes
        [variables,fval,ExitFlag] = fminlbfgs(@(variables) useFun(variables,c_vars,trainX,LabelEqualsPrototype,model.theta,regularization,0,1,dim, costWeight),variables,options);
    end
end
% model.theta = theta;
% model.c_w = c_vars(~isnan(c_vars));
model.w = variables(1:length(model.c_w),2:end);
model.A = variables(length(model.c_w)+1:end,2:end);
model.A=model.A/sqrt(trace(model.A'*model.A));

%%% output of the training
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
            estimatedLabels = p_angleGMLVQ_classify(trainX, model);
            trainError = mean( trainLab ~= estimatedLabels );
			varargout(k) = {trainError};
		case(2)
            testSet = p.Results.testSet;
            if ~isempty(testSet)                
                estimatedLabels = p_angleGMLVQ_classify(testSet(:,1:end-1), model);
                testError = mean( testSet(:,end) ~= estimatedLabels );
                varargout(k) = {testError};
            else
                warning('Test error was requested, yet no testSet provided!');
            end			
        case(3)
			varargout(k) = {fval};
        case(4)
            varargout(k) = {p.Results};
	end
end
end

