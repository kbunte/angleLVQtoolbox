function [f, G]  = cw_angleLGMLVQ_optfun(variables,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim,costWeights)
% [f G] = GMLVQ_optfun(variables) 
%this function optimizes teh cost functions wrt the prototype and the
%matrix. When Lprototypes=1 then optimization of prototype occur. When 
% Lrelevances is 1 then optimization of matrix occur. Lprototypes and
% Lrelevances are not simultaneously zero or one; The variable calle
% variables contain the prototypes w along with labels in its first n_w
% rows (where n_w= no.of prototypes per class*no.of classes). The first 
% column of these rows contain the labels of the classes of your dataset. 
% The remaining rows contain the relevance matrix. The first column 
% corresponding to these rows are filled with NaNs.The remaining
% rows contain the matrix corresponding to each prototype. This script is
% corresponding to 1 prototype per class. If your class labels are not
% regular (i.e., instead of 1,2,3,4... suppose you have 1,5,8,9,etc) then
% please tweak line 86 such that all your class labels are considered.

% save('test');error('load test')
if isempty(beta)
    ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
end
useX = trainX;useX(isnan(trainX)) = 0;

relIdx = find(isnan(variables(:,1)));
wIdx= 1:size(variables,1);wIdx(relIdx) = [];
w   = variables(wIdx,2:end);
c_w = variables(wIdx,1);
if length(c_w)~=length(dim)
    c_A = unique(round(c_w));classwise = 1;
else
    c_A = c_w;classwise = 0;
end
Aidx = mat2cell(1:length(relIdx),1,dim);
A = variables(relIdx,2:end);
xAw = nan(size(trainX,1),length(c_w));
normxA = nan(size(trainX,1),length(c_w));
normwA = nan(length(c_w),1);
cosa = nan(size(trainX,1),length(c_w));
for j=1:length(c_w)
    if classwise
        useAidx = Aidx{c_A==c_w(j)};
    else
        useAidx = Aidx{j};
    end
    xAw(:,j) = useX*A(useAidx,:)'*A(useAidx,:)*w(j,:)';
    normwA(j) = sqrt(diag(w(j,:)*A(useAidx,:)'*A(useAidx,:)*w(j,:)'));        
    normxA(:,j) = sqrt(diag(useX*A(useAidx,:)'*A(useAidx,:)*useX'));        
    cosa(:,j) = bsxfun(@rdivide,xAw(:,j),normxA(:,j)*normwA(j)');
end
dists = ca2d(cosa,beta);

nb_prototypes = length(c_w);

Dwrong = dists;
Dwrong(LabelEqualsPrototype) = realmax(class(Dwrong));   % set correct labels impossible
[distwrong, pidxwrong] = min(Dwrong.'); % closest wrong
clear Dwrong;

Dcorrect = dists;
Dcorrect(~LabelEqualsPrototype) = realmax(class(Dcorrect)); % set wrong labels impossible
[distcorrect, pidxcorrect] = min(Dcorrect.'); % closest correct
clear Dcorrect;

distcorrectpluswrong = distcorrect + distwrong;
distcorrectminuswrong= distcorrect - distwrong;

mu = distcorrectminuswrong ./ distcorrectpluswrong;
regTerm = 0;
if regularization
%     regTerm = regularization * log(det(A*A'));
    for j=1:length(Aidx)
        regTerm = regTerm + regularization * log(det(A(Aidx{j},:)*A(Aidx{j},:)'));
    end
end
% normTerm = 0;
% for j=1:length(Aidx)
%     normTerm = normTerm + (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),Aidx{j})))^2;
% end
[~,estidx] = min(dists,[],2);
% nbPerClass=cell2mat(arrayfun(@(c) sum(trainLab==c_A(c)),1:length(c_A),'uni',0)');
nbPerClass = sum(LabelEqualsPrototype);
nb_samples = size(LabelEqualsPrototype,1);
train_cIdx = arrayfun(@(x) find(LabelEqualsPrototype(x,:)==1),1:nb_samples)';
costPerSample = diag(costWeights(train_cIdx,estidx))'.*min(nbPerClass)./nbPerClass(train_cIdx);
% for i=1:nb_samples
%     prot_est=estidx(i);% if prot_est>1, prot_est=prot_est-5; end
%     c=trainLab(i); if c>1, c=c-1; end %uncomment this part only when using GCMS dataset
%     costPerSample(i)=costWeights(c,prot_est)*1/nbPerClass(c);
% end
f = mean(costPerSample.*mu)- regTerm; % + normTerm; % 1/nb_samples*

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
    % update all prototypes
    if Lprototypes
        for k=1:nb_prototypes
            actNW3 = normwA(k)^3;
            idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
            idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong            
            if classwise, useAidx = Aidx{c_A==c_w(k)};else useAidx = Aidx{k};end
            
            xdiffw = useX(idxc,:)*A(useAidx,:)'*A(useAidx,:) .* normwA(k)^2 - bsxfun(@times,xAw(idxc,k) , w(k,:)*A(useAidx,:)'*A(useAidx,:));
            gwj = sum(costPerSample(idxc)'.*bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc,k)*actNW3)));

            xdiffw = useX(idxw,:)*A(useAidx,:)'*A(useAidx,:) .* normwA(k)^2 - bsxfun(@times,xAw(idxw,k) , w(k,:)*A(useAidx,:)'*A(useAidx,:));
            gwk = sum(costPerSample(idxw)'.*bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw,k)*actNW3))); 
            G(k,2:end) = gwj+gwk; % 1/nb_samples.*
        end
    end
    % update relevance vector
    if Lrelevances
        ga = zeros(size(A));
        for d=1:size(A,1)
            actIdx = find(arrayfun(@(x) length(find(d==Aidx{x})),1:length(Aidx)));
            prots = find(c_A(actIdx)==c_w);
            for pidx = 1:length(prots)
                actwJ = find(pidxcorrect==prots(pidx));
                actwK = find(pidxwrong==prots(pidx));
dcosaJdA = bsxfun(@rdivide, bsxfun(@times,trainX(actwJ,:),w(prots(pidx),:)*A(d,:)') + ...
                        bsxfun(@times,w(prots(pidx),:),useX(actwJ,:)*A(d,:)'), normxA(actwJ,prots(pidx)).*normwA(prots(pidx)) ) - ...
              bsxfun(@times, arrayfun(@(j) xAw(actwJ(j),prots(pidx)),1:length(actwJ))' , (...
                bsxfun(@rdivide, bsxfun(@times,trainX(actwJ,:),useX(actwJ,:)*A(d,:)'), normxA(actwJ,prots(pidx)).^3.*normwA(prots(pidx))) + ...
                bsxfun(@rdivide, bsxfun(@times,w(prots(pidx),:),w(prots(pidx),:)*A(d,:)'),normxA(actwJ,prots(pidx)).*normwA(prots(pidx)).^3)  ));
dcosaKdA = bsxfun(@rdivide, bsxfun(@times,trainX(actwK,:),w(prots(pidx),:)*A(d,:)') + ...
                        bsxfun(@times,w(prots(pidx),:),useX(actwK,:)*A(d,:)'), normxA(actwK,prots(pidx)).*normwA(prots(pidx)) ) - ...
              bsxfun(@times, arrayfun(@(j) xAw(actwK(j),prots(pidx)),1:length(actwK))' , (...
                bsxfun(@rdivide, bsxfun(@times,trainX(actwK,:),useX(actwK,:)*A(d,:)'), normxA(actwK,prots(pidx)).^3.*normwA(prots(pidx))) + ...
                bsxfun(@rdivide, bsxfun(@times,w(prots(pidx),:),w(prots(pidx),:)*A(d,:)'),normxA(actwK,prots(pidx)).*normwA(prots(pidx)).^3)  ));            
                %ga(d,:) = ga(d,:) + nansum(costPerSample(actwJ)'.*( bsxfun(@times,(mudJ(actwJ).*dgwJ(actwJ))',dcosaJdA) ) + nansum( bsxfun(@times,(mudK(actwK).*dgwK(actwK))',dcosaKdA)));
            ga(d,:) = ga(d,:) + (nansum( costPerSample(actwJ)'.*bsxfun(@times,(mudJ(actwJ).*dgwJ(actwJ))',dcosaJdA) ) + nansum(costPerSample(actwK)'.* bsxfun(@times,(mudK(actwK).*dgwK(actwK))',dcosaKdA)));
            end
        end
        regTerm = zeros(size(A));
        normF3 = zeros(size(A));
        for j=1:length(Aidx) 
            normF3(Aidx{j},:) = -4*(1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),Aidx{j}))).*A(Aidx{j},:);
%             normF3(Aidx{j},:) = -4*(1-sum(diag(A(Aidx{j},:)'*A(Aidx{j},:)))).*A(Aidx{j},:);
        end
        if regularization
            for j=1:length(Aidx)
                regTerm(Aidx{j},:) = regularization.*2.*(pinv(A(Aidx{j},:)))';
            end
        end
%         G(relIdx,2:end) = ga - min(min(costWeights))*regTerm + min(min(costWeights))*normF3;
        G(relIdx,2:end) = ga - regTerm + normF3; % 1/nb_samples.*
    end
%     G = 1/nb_samples.*G;
%     end
end
end