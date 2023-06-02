function [f G]  = angleLGMLVQ_optfun(variables,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim)
% [f G] = GMLVQ_optfun(variables) 
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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
if length(c_w)~=length(dim) % the c_A is redundant info and can be worked out
    c_A = unique(round(c_w));
    classwise = 1;
else
    c_A = c_w;
    classwise = 0;
end
Aidx = mat2cell(1:length(relIdx),1,dim);
A = variables(relIdx,2:end);
xAw = nan(size(trainX,1),length(c_w));
normxA = nan(size(trainX,1),length(c_w));
normwA = nan(length(c_w),1);
cosa = nan(size(trainX,1),length(c_w));
for j=1:length(c_w)
    if classwise, 
        useAidx = Aidx{c_A==round(c_w(j))};
    else
        useAidx = Aidx{j};
    end
    xAw(:,j) = useX*(A(useAidx,:)'*A(useAidx,:))*w(j,:)';
    normwA(j) = sqrt(diag(w(j,:)*(A(useAidx,:)'*A(useAidx,:))*w(j,:)'));        
    normxA(:,j) = sqrt(diag(useX*(A(useAidx,:)'*A(useAidx,:))*useX'));        
    cosa(:,j) = bsxfun(@rdivide,xAw(:,j),normxA(:,j)*normwA(j)');
end
dists = ca2d(cosa,beta);
[r,~]=find(cosa==1);
for idxR=1:length(r)
    cosa(r(idxR),find(cosa(r(idxR),:)==1))=0.9988;
end
nb_prototypes = length(c_w);

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
if regularization
%     regTerm = regularization * log(det(A*A'));
    for j=1:length(Aidx)
        regTerm = regTerm + regularization * log(det(A(Aidx{j},:)*A(Aidx{j},:)'));
    end
end
normTerm = 0;
for j=1:length(Aidx)
    normTerm = normTerm + (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),Aidx{j})))^2;
end
f = sum(mu) -regTerm + normTerm;
% x0 = A(:);
% g = Grad(@(x0) (1-sum(diag(reshape(x0,length(x0)/size(trainX,2),size(trainX,2))'*reshape(x0,length(x0)/size(trainX,2),size(trainX,2)))))^2, x0);
% test = reshape(g,length(g)/size(trainX,2),size(trainX,2))
% g = Grad(@(x0) log(det(reshape(x0,length(x0)/size(trainX,2),size(trainX,2))*reshape(x0,length(x0)/size(trainX,2),size(trainX,2))')), x0);
% test = reshape(g,length(g)/size(trainX,2),size(trainX,2))
% [test;2.*(pinv(A))']
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
            gwj = sum(bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc,k)*actNW3)));

            xdiffw = useX(idxw,:)*A(useAidx,:)'*A(useAidx,:) .* normwA(k)^2 - bsxfun(@times,xAw(idxw,k) , w(k,:)*A(useAidx,:)'*A(useAidx,:));
            gwk = sum(bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw,k)*actNW3)));
            G(k,2:end) = gwj+gwk;
        end
    end
    % update relevance vector
    if Lrelevances,
        ga = zeros(size(A));
        for d=1:size(A,1)
            actIdx = find(arrayfun(@(x) length(find(d==Aidx{x})),1:length(Aidx)));
            prots = find(c_A(actIdx)==c_w);
            for pidx = 1:length(prots)
                actwJ = find(pidxcorrect==prots(pidx));
                actwK = find(pidxwrong==prots(pidx));
            dcosaJdA =  bsxfun(@rdivide, bsxfun(@times,trainX(actwJ,:),w(prots(pidx),:)*A(d,:)') + ...
                        bsxfun(@times,w(prots(pidx),:),useX(actwJ,:)*A(d,:)'), normxA(actwJ,prots(pidx)).*normwA(prots(pidx)) ) - ...
                        bsxfun(@times, arrayfun(@(j) xAw(actwJ(j),prots(pidx)),1:length(actwJ))' , (...
                        bsxfun(@rdivide, bsxfun(@times,trainX(actwJ,:),useX(actwJ,:)*A(d,:)'), normxA(actwJ,prots(pidx)).^3.*normwA(prots(pidx))) + ...
                        bsxfun(@rdivide, bsxfun(@times,w(prots(pidx),:),w(prots(pidx),:)*A(d,:)'),normxA(actwJ,prots(pidx)).*normwA(prots(pidx)).^3) ));
           
           dcosaKdA =  bsxfun(@rdivide, bsxfun(@times,trainX(actwK,:),w(prots(pidx),:)*A(d,:)') + ...
                        bsxfun(@times,w(prots(pidx),:),useX(actwK,:)*A(d,:)'), normxA(actwK,prots(pidx)).*normwA(prots(pidx)) ) - ...
                        bsxfun(@times, arrayfun(@(j) xAw(actwK(j),prots(pidx)),1:length(actwK))' , (...
                        bsxfun(@rdivide, bsxfun(@times,trainX(actwK,:),useX(actwK,:)*A(d,:)'), normxA(actwK,prots(pidx)).^3.*normwA(prots(pidx))) + ...
                        bsxfun(@rdivide, bsxfun(@times,w(prots(pidx),:),w(prots(pidx),:)*A(d,:)'),normxA(actwK,prots(pidx)).*normwA(prots(pidx)).^3)  ));
            
                ga(d,:) = ga(d,:) + nansum( bsxfun(@times,(mudJ(actwJ).*dgwJ(actwJ))',dcosaJdA) ) + nansum( bsxfun(@times,(mudK(actwK).*dgwK(actwK))',dcosaKdA) );
            end
        end
        regTerm = zeros(size(A));
        normF3 = zeros(size(A));
        for j=1:length(Aidx) 
            normF3(Aidx{j},:) = -4*(1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),Aidx{j}))).*A(Aidx{j},:);
%             normF3(Aidx{j},:) = -4*(1-sum(diag(A(Aidx{j},:)'*A(Aidx{j},:)))).*A(Aidx{j},:);
        end
        if regularization,
            for j=1:length(Aidx)
                regTerm(Aidx{j},:) = regularization.*2.*(pinv(A(Aidx{j},:)))';
            end
        end
        G(relIdx,2:end) = ga + normF3 - regTerm;
    end
%     end
end
end