 % function [f G]  = p_angleGMLVQ_optfun(variables,c_vars,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim)
function [H G]  = p_angleGMLVQ_optfun2(variables,c_vars,trainX,LabelEqualsPrototype,theta,regularization,Lprototypes,Lrelevances,dim)
% see reference https://link.springer.com/chapter/10.1007%2F978-3-319-91253-0_67
% [f G] = GMLVQ_optfun(variables) 
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if isempty(theta)
    error('empty beta value (linear version) of probabilistic KL angle LVQ has not been implemented yet!');
%     ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2P = @(cosa,beta) (exp( beta*cosa+beta)-1)/(exp(2*beta)-1);%     ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
end
useX = trainX;useX(isnan(trainX)) = 0;
nb_features = size(trainX,2);
relIdx = find(isnan(c_vars));
if relIdx
    wIdx= 1:size(c_vars,1);wIdx(relIdx) = [];
    w   = variables(wIdx,2:end);%     c_w = variables(wIdx,1);
    if length(relIdx)>1
        A = variables(relIdx,2:end);
    else
        A = diag(variables(relIdx,:));% diag(sqrt(variables(relIdx,2:end)));
    end
    xAw = useX*A'*A*w';
    normxA = sqrt(diag(useX*A'*A*useX'));
    normwA = sqrt(diag(w*A'*A*w'));
    cosa = bsxfun(@rdivide,xAw,normxA*normwA');
    eD = ca2P(cosa,theta);
else
    error('No relevances not implemented!')
    % TODO
%     w = variables;%     c_w = variables(:,1);
%     xtw = useX*w';
%     normx = sqrt(sum(useX.^2,2));
%     normw = sqrt(sum(w.^2,2))';
%     cosa = bsxfun(@rdivide,xtw,normx*normw);
%     dists = ca2d(cosa,theta);
end
nb_prototypes = size(w,1);
if(1)
    p_x_given_w = eD./(sum(eD,2));%     g_theta = ca2P(cosa,theta);
    normTerm = (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),1:length(relIdx))))^2;
    if isempty(find(isinf(log(p_x_given_w)), 1))
        H = 1/size(LabelEqualsPrototype,1)*sum(-sum(LabelEqualsPrototype.*log(p_x_given_w),2)) + normTerm;
        % because log 1 will be zero 
    else
        fixInf = p_x_given_w;
        [rows, ~] = find(p_x_given_w==0);
        fixInf(rows,:) = fixInf(rows,:)+eps;
        H = 1/size(LabelEqualsPrototype,1)*sum(-sum(LabelEqualsPrototype.*log(p_x_given_w+eps),2)) + normTerm;
    end
end
if(1) % this is the new gradient
%% test the gradient
% TODO: regterm not implemented!
% TODO: not yet adapted for missing values in the gradient
    G = zeros(size(variables)); % initially no gradient
    class_sum_g_theta = sum(eD,2);%         class_sum_g_theta = sum(g_theta,2);
    if isempty(find(isinf(1./(p_x_given_w)), 1))
        % below can be directly be interpret as D_KullbackLeibler(p_A(x)||p(x)):        
        all_diff_KL_factors = 1/size(LabelEqualsPrototype,1).*...
            (-LabelEqualsPrototype./p_x_given_w)./(class_sum_g_theta.^2);
    else
        % TODO: not tested!
        all_diff_KL_factors = 1/size(LabelEqualsPrototype,1).*...
            (-LabelEqualsPrototype./fixInf)./(class_sum_g_theta.^2);
    end
    if isempty(theta)
        % not implemented, does not work that well anyway
        error('empty beta value (linear version) of probabilistic KL angle LVQ has not been implemented yet!');
    else
        all_diff_gFactors = (theta*exp( theta*cosa+theta))/(exp(2*theta)-1);
    end
    if isempty(relIdx) % no relevances only prototypes
        error('empty relevance matrix version of probabilistic KL angle LVQ has not been implemented yet!'); % maybe anyway in its own angleGLVQ instead
    else % using relevances
        if Lprototypes               
            diff_H2w = zeros(nb_prototypes,nb_features);
             for k=1:nb_prototypes
                 dcosa2wk2= 1./(normxA.*normwA(k)^3).*( (w(k,:)*A')*A*w(k,:)'.*(useX*A')*A - xAw(:,k).*(w(k,:)*A')*A );

                for c=1:nb_prototypes                                                   
                        if k==c
                            term2 = all_diff_gFactors(:,c).* dcosa2wk2 .* class_sum_g_theta;
                        else
                            term2=zeros(size(trainX));
                        end
                        diff_H2w(k,:) = diff_H2w(k,:)+ sum(all_diff_KL_factors(:,c).*( term2 - all_diff_gFactors(:,k).* dcosa2wk2 .*eD(:,c) ));
%                             diff_H2w(k,:) = diff_H2w(k,:)+ sum(all_diff_KL_factors(:,c).*( term2 - all_diff_gFactors(:,k).* dcosa2wk2 .*g_theta(:,c) ));
                end
             end

            G(~isnan(c_vars),2:end) = diff_H2w;
        end
        % update relevance vector
        if Lrelevances
            diff_H2A = zeros(size(A)); 
            sumA_xi_x2 = sum(A*useX');  
            all_diffTerm2=zeros(nb_features,nb_prototypes);
            for i=1:size(LabelEqualsPrototype,1)
                for j=1:nb_prototypes
                    sumA_xi_w = sum(A*w(j,:)'); 
                    temp1=(useX(i,:)'*sumA_xi_w' + (w(j,:)*sumA_xi_x2(:,i)')')/(normxA(i)*normwA(j))-...
                        xAw(i,j).*( (useX(i,:)'*sumA_xi_x2(:,i))/((normxA(i)^3)*normwA(j)) + (w(j,:)*sumA_xi_w/(normxA(i)*normwA(j).^3))');
                            temp1(isnan(trainX(i,:)))=0;
                          all_diffTerm2(:,j)=(all_diff_gFactors(i,j) .*temp1);  
                end
%                    unified=(all_diff_KL_factors(i,:))'.*(bsxfun(@times,all_diffTerm2,class_sum_g_theta(i))'-(g_theta(i,:).*(all_diffTerm2))');
                unified=(all_diff_KL_factors(i,:))'.*(bsxfun(@times,all_diffTerm2,class_sum_g_theta(i))'-(eD(i,:).*(all_diffTerm2))');
                diff_H2A=diff_H2A+sum(unified);
            end
            diff_normterm = -4.*( 1-sum(arrayfun(@(d) sum(A(d,:).^2),1:length(relIdx))) ).* A;
            G(isnan(c_vars),2:end) = diff_H2A + diff_normterm;

        end
    end
end
end