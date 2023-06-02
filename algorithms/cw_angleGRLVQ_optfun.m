function [f G]  = cw_angleGRLVQ_optfun(variables,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim,costWeights) % , trainLab
% [f G] = cw_angleGRLVQ_optfun(variables) optimization function for
% classwise angle GRLVQ. 
%returns the cost function value and the Gradient for usage in fminlbfgs
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
c_A = unique(round(c_w));
nb_prototypes = size(w,1);
A = diag(variables(relIdx,2:end));
xAw = useX*A'*A*w';
normxA = sqrt(diag(useX*A'*A*useX'));
normwA = sqrt(diag(w*A'*A*w'));
cosa = bsxfun(@rdivide,xAw,normxA*normwA');
dists = ca2d(cosa,beta);

[~,idx] = min(dists,[],2);
% estimatedLabels = c_w(idx);

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
if regularization, regTerm = regularization * sum(log(diag(A).^2)); end
normTerm = (1-sum(diag(A).^2))^2;

% costWeights = ones(4,4)./sum(sum(ones(4,4)));
% costWeights = [3,2,2,2;2,3,1,1;2,1,3,1;2,1,1,3]./sum(sum([3,2,2,2;2,3,1,1;2,1,3,1;2,1,1,3]));
% costWeights = costWeights./sum(sum(costWeights));

[~,estidx] = min(dists,[],2);
% TODO: this is for now 1 prototype per class THROUGHOUT THE CODE
% classFactor = sum(bsxfun(@rdivide,LabelEqualsPrototype,sum(LabelEqualsPrototype)),2);

% predFactor = zeros(length(mu),1);
% for c=1:size(w,1)
%     predFactor = predFactor + sum(cell2mat(arrayfun(@(p)  (LabelEqualsPrototype(:,c) & estidx==p).*costWeights(c,p) ,1:size(w,1),'uni',0)),2);
% end
% f = sum(classFactor'.*predFactor'.*mu)- regTerm + normTerm;
% costConfusion = zeros(size(w,1));
% for c=1:size(w,1)
%     classIdx = find(LabelEqualsPrototype(:,c)>0);
% %     [LabelEqualsPrototype(classIdx,1),(distcorrect(classIdx)-distwrong(classIdx))',pidxwrong(classIdx)',estidx(classIdx)]
%     costConfusion(c,:) = 1/nbPerClass(c).*arrayfun(@(p)  sum(mu(classIdx(estidx(classIdx)==p))),1:size(w,1));
% end
% f = sum(sum(costWeights.*costConfusion)) - regTerm + normTerm;

% f = sum(costWeights(1,1).*mu) - regTerm + normTerm;
trainLab = c_w(arrayfun(@(r) find(LabelEqualsPrototype(r,:)==1),1:size(LabelEqualsPrototype,1)));
nbPerClass=cell2mat(arrayfun(@(c) sum(trainLab==c_A(c)),1:length(c_A),'uni',0)');
nb_samples=length(trainLab);
for i=1:nb_samples
    prot_est=estidx(i);% if prot_est>1, prot_est=prot_est-5; end
    c=trainLab(i); if c>1, c=c-1; end %uncomment this part only when using GCMS dataset
    costPerSample(i)=costWeights(c,prot_est)*1/nbPerClass(c);
end
f = sum(costPerSample.*mu)- regTerm + normTerm;
%f = sum(classFactor'.*predFactor.*mu)- regTerm + normTerm;
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
    % update prototypes
    if Lprototypes
        for k=1:nb_prototypes % update all prototypes
            actNW3 = normwA(k)^3;
            idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
            idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong
            xdiffw = bsxfun(@times,diag(A)'.^2,trainX(idxc,:).*normwA(k)^2 - bsxfun(@times,xAw(idxc,k),w(k,:)));
            xdiffw(isnan(trainX(idxc,:))) = 0;
            %gwj = sum( classFactor(idxc).*costPerSample(idxc)'.*bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));
            gwj = sum(costPerSample(idxc)'.*bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));
            
            xdiffw = bsxfun(@times,diag(A)'.^2,trainX(idxw,:).*normwA(k)^2 - bsxfun(@times,xAw(idxw,k),w(k,:)));
            xdiffw(isnan(trainX(idxw,:))) = 0;
%            gwk = sum(classFactor(idxw).*costPerSample(idxw)'.*bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));
            gwk = sum(costPerSample(idxw)'.*bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));

            G(k,2:end) = gwj+gwk;
        end
    end
    % update relevance vector
    if Lrelevances        
        dcosaJdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX,normwA(pidxcorrect).^2),bsxfun(@times,w(pidxcorrect,:),normxA.^2)) - ...
            bsxfun(@times,arrayfun(@(c) xAw(c,pidxcorrect(c)),1:length(pidxcorrect))',bsxfun(@times,trainX.^2,normwA(pidxcorrect).^2)+bsxfun(@times,w(pidxcorrect,:).^2,normxA.^2)), ...
            normxA.^3.*normwA(pidxcorrect).^3),diag(A)');

        dcosaKdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX,normwA(pidxwrong).^2),bsxfun(@times,w(pidxwrong,:),normxA.^2)) - ...
            bsxfun(@times,arrayfun(@(c) xAw(c,pidxwrong(c)),1:length(pidxwrong))',bsxfun(@times,trainX.^2,normwA(pidxwrong).^2)+bsxfun(@times,w(pidxwrong,:).^2,normxA.^2)), ...
            normxA.^3.*normwA(pidxwrong).^3),diag(A)');
%         normF3 = zeros(length(relIdx),size(trainX,2));
        normF3 = -4*(1-sum(diag(A).^2)).*diag(A)';
        f3 = zeros(length(relIdx),size(trainX,2));
        if regularization
            f3 =  regularization*(2./(diag(A).^2).*diag(A))';
        end
       % ga = nansum( classFactor.*costPerSample'.* (bsxfun(@times,(mudJ.*dgwJ)',dcosaJdA) + bsxfun(@times,(mudK.*dgwK)',dcosaKdA))) -f3;
       ga = nansum(costPerSample'.* (bsxfun(@times,(mudJ.*dgwJ)',dcosaJdA) + bsxfun(@times,(mudK.*dgwK)',dcosaKdA))) -f3;
        G(relIdx,2:end) = ga - min(min(costWeights))*regTerm + min(min(costWeights))*normF3;% - sqrt(sum(ga.^2)).*diag(A)';
        
%         normF3
    end
end
end