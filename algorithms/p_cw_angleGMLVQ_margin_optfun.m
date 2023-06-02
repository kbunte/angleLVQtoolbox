function [H G]= p_cw_angleGMLVQ_margin_optfun(variables,c_vars,trainX,LabelEqualsPrototype,theta,regularization,Lprototypes,Lrelevances,dim,costWeight)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% References: TODO https://link.springer.com/chapter/10.1007%2F978-3-319-91253-0_67 for Villman Cross-Entropy GMLVQ
if isempty(theta)
    error('empty beta value (linear version) of probabilistic KL angle LVQ has not been implemented yet!')
%     ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2P = @(cosa,beta) (exp( beta*cosa+beta)-1)/(exp(2*beta)-1); %     ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
end
useX = trainX;useX(isnan(trainX)) = 0;
nb_features = size(trainX,2);

relIdx = find(isnan(c_vars));
if relIdx
    wIdx= 1:size(c_vars,1);wIdx(relIdx) = [];
%     c_A = unique(round(c_w));
    w   = variables(wIdx,2:end);
%     c_w = variables(wIdx,1);
    if length(relIdx)>1
        A = variables(relIdx,2:end);
    else
         A = diag(variables(relIdx,2:end)); % diag(sqrt(variables(relIdx,2:end)));
    end
    xAw = (useX*A')*A*w';
    xAx = (useX*A')*A*useX';
    wAw = (w*A')*A*w';
    normxA = sqrt(diag(xAx));
    normwA = sqrt(diag(wAw));
    cosa = bsxfun(@rdivide,xAw,normxA*normwA');
    g_beta = ca2P(cosa,theta);    %     dists = ca2d(cosa,beta);
%     beta=10;fplot(@(cosa) (exp(beta*cosa+beta)-1)/(exp(2*beta)-1), [-1,1]);xlabel('cos a');ylabel('prob');
end
nb_prototypes = size(w,1);
nb_samples=size(LabelEqualsPrototype,1);
if(1)
    class_sum_g_beta = sum(g_beta,2);
    p_x_given_w = g_beta./(class_sum_g_beta);  
    [~,estidx] = max( p_x_given_w,[],2); 
   
    train_cIdx = arrayfun(@(x) find(LabelEqualsPrototype(x,:)==1),1:nb_samples)';
    nbPerClass = sum(LabelEqualsPrototype);
%     The following is normalized, such that the weight is 1 for a sample of the minority class that with maximal costs
%     (min(nbPerClass2)./nbPerClass2(ones(nbPerClass2(c),1).*c)),1:3,'UniformOutput',false))
    costPerSample = diag(costWeight(train_cIdx,estidx)).*min(nbPerClass)./nbPerClass(train_cIdx)'; % arrayfun(@(x) costWeight(costCage(x),estidx(x)),1:nb_samples);
    
    regTerm = 0;
    normTerm = 0;
    if exist('A','var')
        if regularization
            if length(relIdx)==1
                regTerm = regularization * sum(log(-diag(A).^2));
            else
                regTerm = regularization * log(det(A*A'));
            end
        end
        %     normTerm = (1-sum(diag(A'*A)))^2;
        normTerm = ( 1-sum(arrayfun(@(d) sum(A(d,:).^2),1:dim)) )^2;
    end

    % Entropy over probabilities favors high probabilities of one class over close probabilities of several classes:
    % below can be directly be interpret as D_KullbackLeibler(q_A(x)||p(x)):   
    if isempty(find(isinf(log(p_x_given_w)), 1))     
        H = 1/nb_samples*nansum(costPerSample.*(sum(p_x_given_w.*log(p_x_given_w),2)...
                                                                 -sum(p_x_given_w.*log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps),2))) ...
                                                                 + normTerm-regTerm;
    else
        fixInf = p_x_given_w;
        [rows, ~] = find(p_x_given_w==0);
        fixInf(rows,:) = fixInf(rows,:)+eps;
        H = 1/nb_samples*nansum(costPerSample.*(sum(p_x_given_w.*log(fixInf),2)...
                                                                 -sum(p_x_given_w.*log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps),2))) ...
                                                                 + normTerm-regTerm;
    end
end

if(1) % this is the new gradient
%% test the gradient
% TODO: not yet adapted for missing values in the gradient
    G = zeros(size(variables)); % initially no gradient
    if isempty(find(isinf(log(p_x_given_w)), 1))
        % below can be directly be interpret as D_KullbackLeibler(p_A(x)||p(x)):        
        all_diff_KL_factors = (1/nb_samples.*...
            (log(p_x_given_w)-log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps)+1)./(class_sum_g_beta.^2));
    else
        % TODO: not tested!
        all_diff_KL_factors = (1/nb_samples.*...
            (log(fixInf)-log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps)+1)./(class_sum_g_beta.^2));
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
                            term2 = all_diff_gFactors(:,c).* dcosa2wk2 .* class_sum_g_beta;
                        else
                            term2=zeros(size(trainX));
                        end
                        diff_H2w(k,:) = diff_H2w(k,:)+ sum(costPerSample.*all_diff_KL_factors(:,c).*( term2 - all_diff_gFactors(:,k).* dcosa2wk2 .*g_beta(:,c) ));
                end
             end
            G(~isnan(c_vars),2:end) = diff_H2w;
        end
        % update relevance vector
        if Lrelevances
            diff_H2A = zeros(size(A)); 
            sumA_xi_x2 = A*useX'; 
            for i=1:nb_samples
                all_diffTerm = zeros(dim,nb_features,nb_prototypes);%                     
                for j=1:nb_prototypes
                    sumA_xi_w = A*w(j,:)'; 
                    for xi=1:size(A,1)
                        temp=nansum([trainX(i,:)'*sumA_xi_w(xi) , (w(j,:)*sumA_xi_x2(xi,i)')'],2)/(normxA(i)*normwA(j))-...
                        xAw(i,j).*nansum( [(trainX(i,:)'*sumA_xi_x2(xi,i))/((normxA(i)^3)*normwA(j)) , (w(j,:)*sumA_xi_w(xi)/(normxA(i)*normwA(j).^3))'],2);
                            temp(isnan(trainX(i,:)))=0;
                            all_diffTerm(xi,:,j)=all_diff_gFactors(i,j) .*temp;
                    end
                end
                unified=reshape(repmat(costPerSample(i)*all_diff_KL_factors(i,:),dim,1),nb_prototypes*dim,1).*(...
                    reshape(permute(all_diffTerm,[1,3,2]),[],nb_features,1).*repmat(class_sum_g_beta(i),dim*nb_prototypes,1)-...
                    reshape(repmat(g_beta(i,:),dim,1),dim*nb_prototypes,1).*repmat(nansum(all_diffTerm,3),nb_prototypes,1));
                diff_H2A=diff_H2A+sum(permute(reshape(unified,dim,nb_prototypes,nb_features),[1,3,2]),3);
            end
            diff_normterm = -4.*( 1-sum(arrayfun(@(d) sum(A(d,:).^2),1:dim)) ).* A;
            if regularization, regTerm = regularization.*2.*(pinv(A))';end
            G(isnan(c_vars),2:end) = diff_H2A + diff_normterm - regTerm;
        end
    end
end
end
%end