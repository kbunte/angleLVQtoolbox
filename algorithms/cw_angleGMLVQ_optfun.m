function [f,G]  = cw_angleGMLVQ_optfun(variables,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim,costWeights) % ,trainLab
% This is the optfun fucntion for the cost-function of cw_ALVQ
%beta :field empty= linear transformation; value between 0 to 1=>
%exponential transformation
%%this function optimizes teh cost functions wrt the prototype and the
%matrix. When Lprototypes=1 then optimization of prototype occur. When 
% Lrelevances is 1 then optimization of matrix occur. Lprototypes and
% Lrelevances are not simultaneously zero or one; The variable called
% variables contain the prototypes w along with labels in its first n_w
% rows (where n_w= no.of classes). The first column of these rows contain
% the labels of the classes of your dataset. The remaining
% rows contain the relevance matrix. The first column corresponding to 
% these rows are filled with NaNs. If your class labels are not
% regular (i.e., instead of 1,2,3,4... suppose you have 1,5,8,9,etc) then
% please tweak line 96 such that all your class labels are considered.
% 
if isempty(beta)
    ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
end
useX = trainX;useX(isnan(trainX)) = 0; % imputing the missing dimensions in training set with zero

relIdx = find(isnan(variables(:,1)));
if relIdx
    wIdx= 1:size(variables,1);wIdx(relIdx) = [];
    w   = variables(wIdx,2:end);
    c_w = variables(wIdx,1);
    c_A = unique(round(c_w));
    if length(relIdx)>1
        A = variables(relIdx,2:end);
    else
        A = diag(variables(relIdx,2:end));% diag(sqrt(variables(relIdx,2:end)));
    end
    % computation of distance between prototypes and presented samples
    xAw = useX*A'*A*w';
    normxA = sqrt(diag(useX*A'*A*useX'));
    normwA = sqrt(diag(w*A'*A*w'));
    cosa = bsxfun(@rdivide,xAw,normxA*normwA');
    dists = ca2d(cosa,beta);
else
    w = variables(:,2:end);%     c_w = variables(:,1);
    xtw = useX*w';
    normx = sqrt(sum(useX.^2,2));
    normw = sqrt(sum(w.^2,2))';
    cosa = bsxfun(@rdivide,xtw,normx*normw);
    dists = ca2d(cosa,beta);
end
nb_prototypes = size(w,1);

Dwrong = dists;
Dwrong(LabelEqualsPrototype) = realmax(class(Dwrong));   % set correct labels impossible
[distwrong pidxwrong] = min(Dwrong.'); % closest wrong
clear Dwrong;

Dcorrect = dists;
Dcorrect(~LabelEqualsPrototype) = realmax(class(Dcorrect)); % set wrong labels impossible
[distcorrect pidxcorrect] = min(Dcorrect.'); % closest correct
clear Dcorrect;

distcorrectpluswrong = distcorrect + distwrong;
distcorrectminuswrong = distcorrect - distwrong;

mu = distcorrectminuswrong ./ distcorrectpluswrong;
regTerm = 0;
normTerm = 0;
if regularization, regTerm = regularization * sum(log(diag(A).^2)); end 
normTerm = (1-sum(diag(A).^2))^2;
[~,estidx] = min(dists,[],2);
% TODO: this is for now 1 prototype per class THROUGHOUT THE CODE
% classFactor = sum(bsxfun(@rdivide,LabelEqualsPrototype,sum(LabelEqualsPrototype)),2);
% 
% predFactor = zeros(length(mu),1);
% for c=1:size(w,1)
%     predFactor = predFactor + sum(cell2mat(arrayfun(@(p)  (LabelEqualsPrototype(:,c) & estidx==p).*costWeights(c,p) ,1:size(w,1),'uni',0)),2);
% end
% f = sum(classFactor'.*predFactor'.*mu);
if exist('A','var')
    if regularization
        if length(relIdx)==1
%             regTerm = regularization * sum(log(-diag(A).^2));
            regTerm = regularization * sum(log(-diag(A).^2));
        else
            regTerm = regularization * log(det(A*A'));
        end
    end
%     normTerm = (1-sum(diag(A'*A)))^2;
    normTerm = (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),1:length(relIdx))))^2;
end
nbPerClass= sum(LabelEqualsPrototype);
%cell2mat(arrayfun(@(c) sum(trainLab==c_A(c)),1:length(c_A),'uni',0));
% nb_samples = length(trainLab);
nb_samples = size(LabelEqualsPrototype,1);
train_cIdx = arrayfun(@(x) find(LabelEqualsPrototype(x,:)==1),1:nb_samples)';
costPerSample = diag(costWeights(train_cIdx,estidx)).*min(nbPerClass)./nbPerClass(train_cIdx)'; 
% f = sum(mu)-regTerm+normTerm;
%nbPerClass=([sum(trainLab==1) sum(trainLab==7) sum(trainLab==8) sum(trainLab==9)]);

% % for i=1:nb_samples
% %     prot_est=estidx(i);% if prot_est>1, prot_est=prot_est-5; end
% %     c=trainLab(i); if c>1, c=c-1; end %uncomment this part only when using GCMS dataset
% %     costPerSample(i)=costWeights(c,prot_est)*1/nbPerClass(c);
% % end
f =mean(costPerSample.*mu')- regTerm + normTerm;
%
if nargout > 1  % gradient needed not just function eval
    G = zeros(size(variables)); % initially no gradient
    if isempty(beta)
        dgwJ = -0.5.*arrayfun(@(x) cosa(x,pidxcorrect(x)),1:length(pidxcorrect));
        dgwK = -0.5.*arrayfun(@(x) cosa(x,pidxwrong(x))  ,1:length(pidxwrong));
    else
        dgwJ = -beta/(exp(2*beta)-1) .* exp(-beta.*arrayfun(@(x) cosa(x,pidxcorrect(x)),1:length(pidxcorrect))+beta);
        dgwK = -beta/(exp(2*beta)-1) .* exp(-beta.*arrayfun(@(x) cosa(x,pidxwrong(x))  ,1:length(pidxwrong))  +beta);
    end
    mudJ =  2*distwrong  ./(distcorrectpluswrong.^2);
    mudK = -2*distcorrect./(distcorrectpluswrong.^2);
    if isempty(relIdx)
        for k=1:nb_prototypes % update all prototypes        
            actNW3 = normw(k)^3;
            idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
            idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong

            xdiffw = trainX(idxc,:).*normw(k)^2 - bsxfun(@times,xtw(idxc,k),w(k,:));
            xdiffw(isnan(trainX(idxc,:))) = 0;
            gwj = sum(costPerSample(idxc)'.*bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));
            xdiffw = trainX(idxw,:).*normw(k)^2 - bsxfun(@times,xtw(idxw,k),w(k,:));
            xdiffw(isnan(trainX(idxw,:))) = 0;
            gwk = sum(costPerSample(idxc)'.*bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));
            
            G(k,2:end) = gwj+gwk;
        end
    else % using relevances
        if Lprototypes
        for k=1:nb_prototypes % update all prototypes        
            actNW3 = normwA(k)^3;
            idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
            idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong
            if length(relIdx)>1
                xdiffw = useX(idxc,:)*A'*A .* normwA(k)^2 - bsxfun(@times,xAw(idxc,k) , w(k,:)*A'*A);
                gwj = sum(costPerSample(idxc).*bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));
                
                xdiffw = useX(idxw,:)*A'*A .* normwA(k)^2 - bsxfun(@times,xAw(idxw,k) , w(k,:)*A'*A);
                gwk = sum(costPerSample(idxw).*bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));         
            else
                xdiffw = bsxfun(@times,diag(A)'.^2,trainX(idxc,:).*normwA(k)^2 - bsxfun(@times,xAw(idxc,k),w(k,:)));
                xdiffw(isnan(trainX(idxc,:))) = 0;
                gwj = sum(costPerSample(idxc).*bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));

                xdiffw = bsxfun(@times,diag(A)'.^2,trainX(idxw,:).*normwA(k)^2 - bsxfun(@times,xAw(idxw,k),w(k,:)));
                xdiffw(isnan(trainX(idxw,:))) = 0;
                gwk = sum(costPerSample(idxw).*bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));

            end
            G(k,2:end) = gwj+gwk;
        end
        end
        % update relevance vector
        if Lrelevances,
            regTerm = zeros(length(relIdx),size(trainX,2));
            normF3 = zeros(length(relIdx),size(trainX,2));
            ga = zeros(size(A)); %f3=0.0001;
            for d=1:size(A,1)
                dcosaJdA = bsxfun(@rdivide,bsxfun(@times,trainX,w(pidxcorrect,:)*A(d,:)') + bsxfun(@times,w(pidxcorrect,:),useX*A(d,:)'),normxA.*normwA(pidxcorrect)) - ...
                        bsxfun(@times,arrayfun(@(j) xAw(j,pidxcorrect(j)),1:length(pidxcorrect))' , ...
                        (bsxfun(@rdivide,bsxfun(@times,trainX,useX*A(d,:)'),normxA.^3.*normwA(pidxcorrect)) + ...
                        bsxfun(@rdivide,bsxfun(@times,w(pidxcorrect,:),w(pidxcorrect,:) *A(d,:)'),normxA.*normwA(pidxcorrect).^3) ));
                dcosaKdA = bsxfun(@rdivide,bsxfun(@times,trainX,w(pidxwrong,:)*A(d,:)') + bsxfun(@times,w(pidxwrong,:),useX*A(d,:)'),normxA.*normwA(pidxwrong)) - ...
                        bsxfun(@times,arrayfun(@(j) xAw(j,pidxwrong(j)),1:length(pidxwrong))' , ...
                        (bsxfun(@rdivide,bsxfun(@times,trainX,useX*A(d,:)'),normxA.^3.*normwA(pidxwrong)) + ...
                        bsxfun(@rdivide,bsxfun(@times,w(pidxwrong,:),w(pidxwrong,:)*A(d,:)'),normxA.*normwA(pidxwrong).^3) ));
%                  if regularization
%                         f3 =  regularization*(2./(diag(A).^2).*diag(A))';
%                  end
                ga(d,:) = nansum(costPerSample.* (bsxfun(@times,(mudJ.*dgwJ)',dcosaJdA) + bsxfun(@times,(mudK.*dgwK)',dcosaKdA)));
            end
            normF3 = -4*(1-sum(diag(A'*A))).*A;
            if regularization,regTerm = regularization.*2.*(pinv(A))';end
            %G(relIdx,2:end) = ga - min(min(costWeights))*regTerm + min(min(costWeights))*normF3;
             G(relIdx,2:end) = ga - regTerm + normF3;
        end
    end
end
end
