[filePath,name,ext] = fileparts(matlab.desktop.editor.getActiveFilename);
addpath(genpath(sprintf('%s',filePath)));
eval(sprintf('cd %s',filePath));
%% load some demo data and prepare Cross Validation
load fisheriris.mat
Y = grp2idx(species);
X = meas; % just look at 3 dimensions to be able to plot the result
figure(1);
gscatter(X(:,2), X(:,3), species,'rgb','osd');

CrossValIdx = cvpartition(Y,'KFold',5);
%% run angleLVQ and local ALVQ models
reps = 5;
beta = 1;
dim = 2;
dims_local = dim*ones(1,3);
prepros = cell(CrossValIdx.NumTestSets,1);
% global ALVQ
ALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
ALVQ = cell(CrossValIdx.NumTestSets,reps);
% local ALVQ
LALVQ = cell(CrossValIdx.NumTestSets,reps);
LALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
for fold=1:CrossValIdx.NumTestSets
    fprintf('processing fold %i\n',fold);
    % z-score transformation preprocessing
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:),"omitmissing"),'S',std(X(CrossValIdx.training(fold),:),"omitmissing"));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=Y(CrossValIdx.training(fold));
    testLab=Y(CrossValIdx.test(fold));
    for iter=1:reps
        rng(fold*10+iter); % for reproducibility and same initialization of all models
        % train the ALVQ model with parameters above. 
        % Check for more options use "help angleGMLVQ_train"
        [actModel,fval] = angleGMLVQ_train(trainX, trainLab,'testSet',[testX,testLab],'beta',beta,'dim',dim,'regularization',0,'Display','off');
        ALVQ{fold,iter} = actModel;
        estTrain=angleGMLVQ_classify(trainX,actModel);trainE= mean(estTrain~=trainLab);% confusionmat(trainLab,estTrain)
        estTest =angleGMLVQ_classify(testX ,actModel);testE = mean(estTest ~=testLab); % confusionmat(testLab,estTest)
        ALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, 1-trainE,1-testE]);

        rng(fold*10+iter); % for reproducibility and same initialization of all localized (more complex) models
        actModel_LALVQ = angleGMLVQ_train(trainX, trainLab,'testSet',[testX,testLab],'beta',beta,'dim',dims_local,'PrototypesPerClass',ones(1,3),'regularization',0,'Display','off');
        LALVQ{fold,iter} = actModel_LALVQ;
        estTrain_local=angleGMLVQ_classify(trainX,actModel_LALVQ); % confusionmat(trainLab,estTrain)
        estTest_local =angleGMLVQ_classify(testX ,actModel_LALVQ); % confusionmat(testLab,estTest)
        LALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, mean(estTrain_local==trainLab),mean(estTest_local ==testLab)]);
    end
end
%% print the performances
fprintf('        ALVQ       LALVQ\nFold Train Test Train Test\n---------------------------\n');
for fold = 1:CrossValIdx.NumTestSets
    fprintf('%4i %5.2f %4.2f %5.2f %4.2f\n',fold,...
        table2array( varfun(@mean,ALVQ_performance(ALVQ_performance.fold==fold,3:end))),...
        table2array( varfun(@mean,LALVQ_performance(LALVQ_performance.fold==fold,3:end))) );
end
fprintf('---------------------------\n all %5.2f %4.2f %5.2f %4.2f\n',...
        table2array( varfun(@mean,ALVQ_performance(:,3:end))),...
        table2array( varfun(@mean,LALVQ_performance(:,3:end))) );
%% plot the global linear discrinative projection
% for visualization check out the sphere of Doom available at git@github.com:SrGh31/classificationSphereMollweide.git
%% demo of the probabilistic version
theta = 1;
pweights = [ 3, 1, 1 ; 
             1, 3, 4 ; 
             1, 2, 3 ];
prepros = cell(CrossValIdx.NumTestSets,1);
% global ALVQ
pALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
pALVQ= cell(CrossValIdx.NumTestSets,reps);
cw_pALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
cw_pALVQ= cell(CrossValIdx.NumTestSets,reps);
for fold=1:CrossValIdx.NumTestSets
    fprintf('processing fold %i\n',fold);
    % z-score transformation preprocessing
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:),"omitmissing"),'S',std(X(CrossValIdx.training(fold),:),"omitmissing"));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=Y(CrossValIdx.training(fold));
    testLab=Y(CrossValIdx.test(fold));
    for iter=1:reps
        rng(fold*10+iter); % for reproducibility and same initialization of all models 
        [actModel_pALVQ,trainE,testE,fval] = p_angleGMLVQ_train(trainX, trainLab,'testSet',[testX,testLab],'theta',theta,'dim',dim,'marginVersion',1,'regularization',0,'Display','off');
        pALVQ{fold,iter}=actModel_pALVQ;
        estTrain_pALVQ = p_angleGMLVQ_classify(trainX,actModel_pALVQ); % confusionmat(trainLab,estTrain_pALVQ)
        estTest_pALVQ  = p_angleGMLVQ_classify(testX ,actModel_pALVQ); % confusionmat(testLab,estTest_pALVQ)
% ALVQ_performance((fold-1)*reps+iter,1:4)
% [mean(estTrain_pALVQ==trainLab),mean(estTest_pALVQ==testLab)]
        pALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, mean(estTrain_pALVQ==trainLab),mean(estTest_pALVQ==testLab)]);

        rng(fold*10+iter); % for reproducibility and same initialization of all models 
        [actModel_pALVQ_cw] = p_angleGMLVQ_train(trainX, trainLab,'costWeight',pweights,'theta',theta,'dim',dim,'marginVersion',1,'regularization',0,'Display','off');
        cw_pALVQ{fold,iter}=actModel_pALVQ_cw;
        estTrain_pALVQ_cw = p_angleGMLVQ_classify(trainX,actModel_pALVQ_cw); % confusionmat(trainLab,estTrain_pALVQ_cw)
        estTest_pALVQ_cw  = p_angleGMLVQ_classify(testX ,actModel_pALVQ_cw); % confusionmat(testLab,estTest_pALVQ_cw)
% ALVQ_performance((fold-1)*reps+iter,1:4)
% [mean(estTrain_pALVQ_cw==trainLab),mean(estTest_pALVQ_cw==testLab)]
        cw_pALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, mean(estTrain_pALVQ_cw==trainLab),mean(estTest_pALVQ_cw==testLab)]);
    end
end
%% print the probabilistic performances
fprintf('        pALVQ    cw_pALVQ\nFold Train Test Train Test\n---------------------------\n');
for fold = 1:CrossValIdx.NumTestSets
    fprintf('%4i %5.2f %4.2f %5.2f %4.2f\n',fold,...
        table2array( varfun(@mean,pALVQ_performance(pALVQ_performance.fold==fold,3:end))),...
        table2array( varfun(@mean,cw_pALVQ_performance(cw_pALVQ_performance.fold==fold,3:end))) );
end
fprintf('---------------------------\n all %5.2f %4.2f %5.2f %4.2f\n',...
        table2array( varfun(@mean,pALVQ_performance(:,3:end))),...
        table2array( varfun(@mean,cw_pALVQ_performance(:,3:end))) );
%% demonstrate the cost weighted version (cw) this is for imbalanced classes so here it does not help
% CXC matrix where C is the number of classes you have, incentivising the
% costfunction to decrease certain missclassification types more than others
weights = [ 4, 1, 1 ; 
            1, 4, 5 ; 
            1, 1, 4 ];
weights_local = [ 10, 1, 1 ; 
                  1, 10, 5 ; 
                  1, 5, 10 ];
prepros = cell(CrossValIdx.NumTestSets,1);
% global ALVQ
cw_ALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
cw_ALVQ= cell(CrossValIdx.NumTestSets,reps);
% local ALVQ
cw_LALVQ_performance = array2table(nan(CrossValIdx.NumTestSets*reps,4),"VariableNames",{'fold','rep','trainAcc','testAcc'});
cw_LALVQ = cell(CrossValIdx.NumTestSets,reps);
for fold=1:CrossValIdx.NumTestSets
    fprintf('processing fold %i\n',fold);
    % z-score transformation preprocessing
    prepros{fold}=struct('M',mean(X(CrossValIdx.training(fold),:),"omitmissing"),'S',std(X(CrossValIdx.training(fold),:),"omitmissing"));
    trainX=bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.training(fold),:),prepros{fold}.M),prepros{fold}.S);
    testX =bsxfun(@rdivide,bsxfun(@minus,X(CrossValIdx.test(fold),:),    prepros{fold}.M),prepros{fold}.S);
    trainLab=Y(CrossValIdx.training(fold));
    testLab=Y(CrossValIdx.test(fold));
    for iter=1:reps
        rng(fold*10+iter); % for reproducibility and same initialization of all models 
        [cw_actModel,~] = cw_angleGMLVQ_train(trainX, trainLab,'beta',beta,'dim',dim,'regularization',0,'costWeight',weights,'PrototypesPerClass',1,'Display','off');
        cw_ALVQ{fold,iter} = cw_actModel;
        cw_estTrain=angleGMLVQ_classify(trainX,cw_actModel); % confusionmat(trainLab,cw_estTrain_local)
        cw_estTest =angleGMLVQ_classify(testX ,cw_actModel); % confusionmat(testLab,cw_estTest_local)
% ALVQ_performance((fold-1)*reps+iter,1:4)
% [mean(cw_estTrain==trainLab),mean(cw_estTest==testLab)]
        cw_ALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, mean(cw_estTrain==trainLab),mean(cw_estTest==testLab)]);
        
        rng(fold*10+iter);
        [cw_actModel_local,costs] = cw_angleGMLVQ_train(trainX, trainLab,'beta',beta,'dim',dims_local,'regularization',0,'costWeight',weights_local,'PrototypesPerClass',1,'Display','off');
        cw_LALVQ{fold,iter} = cw_actModel_local;
        cw_estTrain_local=angleGMLVQ_classify(trainX,cw_actModel_local); % confusionmat(trainLab,cw_estTrain_local)
        cw_estTest_local =angleGMLVQ_classify(testX ,cw_actModel_local); % confusionmat(testLab,cw_estTest_local)
% LALVQ_performance((fold-1)*reps+iter,1:4)
% [mean(cw_estTrain_local==trainLab),mean(cw_estTest_local ==testLab)]
% [confusionmat(trainLab,cw_estTrain_local),confusionmat(testLab,cw_estTest_local)]
        cw_LALVQ_performance((fold-1)*reps+iter,1:4) = array2table([fold, iter, mean(cw_estTrain_local==trainLab),mean(cw_estTest_local ==testLab)]);
    end
end
%% print the performances
fprintf('costweight ALVQ    LALVQ\nFold Train Test Train Test\n---------------------------\n');
for fold = 1:CrossValIdx.NumTestSets
    fprintf('%4i %5.2f %4.2f %5.2f %4.2f\n',fold,...
        table2array( varfun(@mean,cw_ALVQ_performance(ALVQ_performance.fold==fold,3:end))),...
        table2array( varfun(@mean,cw_LALVQ_performance(LALVQ_performance.fold==fold,3:end))) );
end
fprintf('---------------------------\n all %5.2f %4.2f %5.2f %4.2f\n',...
        table2array( varfun(@mean,cw_ALVQ_performance(:,3:end))),...
        table2array( varfun(@mean,cw_LALVQ_performance(:,3:end))) );
