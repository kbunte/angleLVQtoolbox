function [H G]= p_angleGMLVQ_margin_optfun(variables,c_vars,trainX,LabelEqualsPrototype,theta,regularization,Lprototypes,Lrelevances,dim)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% References: TODO https://link.springer.com/chapter/10.1007%2F978-3-319-91253-0_67 for Villman Cross-Entropy GMLVQ
if isempty(theta)
    error('empty theta value (linear version) of probabilistic KL angle LVQ has not been implemented yet!')
%     ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2P = @(cosa,beta) (exp( beta*cosa+beta)-1)/(exp(2*beta)-1);%     ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
    %     beta=10;fplot(@(cosa) (exp(beta*cosa+beta)-1)/(exp(2*beta)-1), [-1,1]);xlabel('cos a');ylabel('prob');
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
        A = diag(variables(relIdx,2:end));% diag(sqrt(variables(relIdx,2:end)));
    end
    xAw = (useX*A')*A*w';
    xAx = (useX*A')*A*useX';
    wAw = (w*A')*A*w';
    normxA = sqrt(diag(xAx));
    normwA = sqrt(diag(wAw));
    cosa = bsxfun(@rdivide,xAw,normxA*normwA');
    g_beta = ca2P(cosa,theta);%     dists = ca2d(cosa,beta);
end
nb_prototypes = size(w,1);
nb_samples=size(LabelEqualsPrototype,1);

p_x_given_w = g_beta./(sum(g_beta,2));   
% [val,idx] = max(p_x_given_w,[],2)
regTerm = 0; % TODO regularization for A'A: not implemented yet!
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
%     fplot(@(p) sum([-p*log(p),-(1-p)*log(1-p)]),[0 1]); 
if isempty(find(isinf(log(p_x_given_w)), 1))
    % below can be directly be interpret as D_KullbackLeibler(p_A(x)||p(x)):        
    H = 1/nb_samples*nansum(sum(p_x_given_w.*log(p_x_given_w),2)...
                           -sum(p_x_given_w.*log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps),2)) + normTerm-regTerm;
else
    fixInf = p_x_given_w;
    [rows, ~] = find(p_x_given_w==0);
    fixInf(rows,:) = fixInf(rows,:)+eps;
    H = 1/nb_samples*nansum(nansum(p_x_given_w.*log(fixInf),2)...
                           -nansum(p_x_given_w.*log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps),2)) + normTerm-regTerm;
end

% lines below for checking the classification error
% c_w = c_vars(~isnan(c_vars));
% [~,trainC]=max(LabelEqualsPrototype,[],2);
% [estLabs,props] = p_angleGMLVQ_classify(trainX,struct('beta',beta,'c_w',c_w,'w',variables(1:length(c_w),2:end),'A',variables(length(c_w)+1:end,2:end)));
% fprintf('Errors after training: train %.3f\n',mean(estLabs~=trainC));

if(1) % this is the new gradient
%% test the gradient
% TODO: not yet adapted for missing values in the gradient
    G = zeros(size(variables)); % initially no gradient
    class_sum_g_beta = sum(g_beta,2);
    if isempty(find(isinf(log(p_x_given_w)), 1))
        % below can be directly be interpret as D_KullbackLeibler(p_A(x)||p(x)):        
        all_diff_KL_factors = 1/nb_samples.*...
            (log(p_x_given_w)-log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps)+1)./(class_sum_g_beta.^2);
    else
        % TODO: not tested!
        all_diff_KL_factors = 1/nb_samples.*...
            (log(fixInf)-log(LabelEqualsPrototype.*(1-nb_prototypes*eps)+eps)+1)./(class_sum_g_beta.^2);
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
%                 xAA = zeros(nb_samples,nb_features);
%                 for i=1:nb_samples,for j=1:nb_features,
%                         for d=1:dim
%                             for xi=1:nb_features
%                                 xAA(i,xi) = xAA(i,xi)+trainX(i,j)*A(d,j)*A(d,xi);
%                             end
%                         end
%                     end
%                 end
%                 test_xAA = (trainX*A')*A;
%                 xAA-test_xAA

%                 wAA = zeros(size(w,1),nb_features);
%                 for i=1:size(w,1)
%                     for j=1:nb_features
%                         for d=1:dim
%                             for xi=1:nb_features
%                                 wAA(i,xi) = wAA(i,xi)+w(i,j)*A(d,j)*A(d,xi);
%                             end
%                         end
%                     end
%                 end
%                 test_wAA = (w*A')*A;
%                 wAA-test_wAA

%                 allx_diff_cosa2w = zeros(nb_samples,nb_features,nb_prototypes);
%                 for c=1:size(w,1)
%                     allx_diff_cosa2w(:,:,c) = 1./(normxA.*normwA(c)).*( (w(c,:)*A')*A*w(c,:)'.*(trainX*A')*A - xAw(:,c).*(w(c,:)*A')*A );
%                 end
%                 test_diff_H2w = squeeze(sum(allx_diff_cosa2w,1))';
            diff_H2w = zeros(nb_prototypes,nb_features);
             for k=1:nb_prototypes
                 dcosa2wk2= 1./(normxA.*normwA(k)^3).*( (w(k,:)*A')*A*w(k,:)'.*(useX*A')*A - xAw(:,k).*(w(k,:)*A')*A );

                for c=1:nb_prototypes                                                   
                        if k==c
                            term2 = all_diff_gFactors(:,c).* dcosa2wk2 .* class_sum_g_beta;
                        else
                            term2=zeros(size(trainX));
                        end
                        diff_H2w(k,:) = diff_H2w(k,:)+ sum(all_diff_KL_factors(:,c).*( term2 - all_diff_gFactors(:,k).* dcosa2wk2 .*g_beta(:,c) ));
                end
             end

            G(~isnan(c_vars),2:end) = diff_H2w;
        end
        % update relevance vector
        if Lrelevances
            diff_H2A = zeros(size(A)); 
            sumA_xi_x2 = A*useX'; 
            for i=1:nb_samples
                all_diffTerm = zeros(dim,nb_features,nb_prototypes);
%                     all_diffTerm=all_diffTerm;
                for j=1:nb_prototypes
                    sumA_xi_w = A*w(j,:)'; 
                    for xi=1:size(A,1)

                        temp=nansum([trainX(i,:)'*sumA_xi_w(xi) , (w(j,:)*sumA_xi_x2(xi,i)')'],2)/(normxA(i)*normwA(j))-...
                        xAw(i,j).*nansum( [(trainX(i,:)'*sumA_xi_x2(xi,i))/((normxA(i)^3)*normwA(j)) , (w(j,:)*sumA_xi_w(xi)/(normxA(i)*normwA(j).^3))'],2);
                            temp(isnan(trainX(i,:)))=0;
                            all_diffTerm(xi,:,j)=all_diff_gFactors(i,j) .*temp;
                    end
                end
                unified=reshape(repmat(all_diff_KL_factors(i,:),dim,1),nb_prototypes*dim,1).*(...
                    reshape(permute(all_diffTerm,[1,3,2]),[],nb_features,1).*repmat(class_sum_g_beta(i),dim*nb_prototypes,1)-...
                    reshape(repmat(g_beta(i,:),dim,1),dim*nb_prototypes,1).*repmat(nansum(all_diffTerm,3),nb_prototypes,1));
                diff_H2A=diff_H2A+sum(permute(reshape(unified,dim,nb_prototypes,nb_features),[1,3,2]),3);
            end
            diff_normterm = -4.*( 1-sum(arrayfun(@(d) sum(A(d,:).^2),1:length(relIdx))) ).* A;
            if regularization, regTerm = regularization.*2.*(pinv(A))';end
            G(isnan(c_vars),2:end) = diff_H2A + diff_normterm - regTerm;
%                 diff_H2A = zeros(size(A));
%                 for i=1:nb_samples
%                     sumA_xi_x = A*useX(i,:)';
%                     all_diffTerm = zeros(size(A,1),size(A,2),nb_prototypes);
%                     for j=1:nb_prototypes
%                         sumA_xi_w = A*w(j,:)';                        
%                         for xi=1:size(A,1)
%                             for eta=1:size(A,2)
%                                 if isnan(trainX(i,eta))
%                                     temp=0; 
%                                 else
%                                     temp=(nansum( [trainX(i,eta)*sumA_xi_w(xi) , w(j,eta)*sumA_xi_x(xi)] )/(normxA(i)*normwA(j)) - ...
%                                   xAw(i,j) * nansum( [(trainX(i,eta)*sumA_xi_x(xi))/(normxA(i)^3*normwA(j)) , (w(j,eta)*sumA_xi_w(xi))/(normxA(i)*normwA(j)^3)]));
%                                 end    
%                                  all_diffTerm(xi,eta,j) = all_diff_gFactors(i,j) .*temp;
%                             end
%                         end
%                     end
% %                     for j=1:nb_prototypes
% %                         all_diffTerm(:,:,j) = all_diff_gFactors(i,j).* (  );
% %                     end
%                     for c=1:nb_prototypes                        
%                         diff_H2A = diff_H2A + all_diff_KL_factors(i,c).*( ...
%                             all_diffTerm(:,:,c) .*class_sum_g_beta(i) - ...
%                             g_beta(i,c) .* nansum(all_diffTerm,3) );
%                     end
%                 end
%                 diff_normterm = -4.*( 1-sum(arrayfun(@(d) sum(A(d,:).^2),1:length(relIdx))) ).* A;
% %                 G(isnan(c_vars),2:end) = diff_H2A;
%                 G(isnan(c_vars),2:end) = diff_H2A + diff_normterm;
        end
    end
end
end
%end