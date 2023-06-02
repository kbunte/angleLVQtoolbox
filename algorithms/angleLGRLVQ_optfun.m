function [f G]  = angleLGRLVQ_optfun(variables,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim)
% [f G] = GMLVQ_optfun(variables) 
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if isempty(beta)
    ca2d = @(cosa,beta) 0.5-0.5*cosa;
else
    ca2d = @(cosa,beta) (exp(-beta*cosa+beta)-1)/(exp(2*beta)-1);
end
useX = trainX;useX(isnan(trainX)) = 0;

relIdx = find(isnan(variables(:,1)));
% if relIdx
wIdx= 1:size(variables,1);wIdx(relIdx) = [];
c_w = variables(wIdx,1);
w   = variables(wIdx,2:end);%     c_w = variables(wIdx,1);
%     if length(relIdx)>1
%         A = variables(relIdx,2:end);
%     else
if length(c_w)~=length(dim), c_A = unique(round(c_w));classwise = 1;else c_A = c_w;classwise = 0;end
Aidx = mat2cell(1:length(relIdx),1,dim);
%allRel=variables(relIdx,2:end);
%A = arrayfun(@(c) diag(allRel(c,:)), 1:size(allRel,1),'uni',0);
A=reshape(variables(relIdx,2:end)',size(trainX,2),length(c_A),[])'; 

for j=1:length(c_w)
    if classwise, useAidx = Aidx{c_A==c_w(j)};else useAidx = Aidx{j};end
    %R(:,:,j)=diag(A(:,useAidx).^2); % 
    R=diag(A(useAidx,:));
    xAw(:,j) = useX*R'*R*w(j,:)'; 
    %xAw(:,j) = useX*A(useAidx,:)'*A(useAidx,:)*w(j,:)';
    normwA(j) =  sqrt(diag(w(j,:)*R'*R*w(j,:)'));
   % normwA(j) =  sqrt(diag(w(j,:)*A(useAidx,:)'*A(j,:)*w(j,:)')) ;    
    normxA(:,j)= sqrt(diag(useX*R'*R*useX'))';   % 
    %normxA(:,j) = sqrt(diag(useX*A(useAidx,:)'*A(useAidx,:)*useX')) ;
    cosa(:,j) = bsxfun(@rdivide,xAw(:,j),normxA(:,j)*normwA(j)');
end
dists = ca2d(cosa,beta);
% else
%     w = variables(:,2:end);%     c_w = variables(:,1);
%     xtw = useX*w';
%     normx = sqrt(sum(useX.^2,2));
%     normw = sqrt(sum(w.^2,2))';
%     cosa = bsxfun(@rdivide,xtw,normx*normw);
%     dists = ca2d(cosa,beta);
% end
nb_prototypes = size(w,1);

Dwrong = dists;
Dwrong(LabelEqualsPrototype) = realmax(class(Dwrong));   % set correct labels impossible
[distwrong pidxwrong] = min(Dwrong.'); % closest wrong
%clear Dwrong;

Dcorrect = dists;
Dcorrect(~LabelEqualsPrototype) = realmax(class(Dcorrect)); % set wrong labels impossible
[distcorrect pidxcorrect] = min(Dcorrect.'); % closest correct
%clear Dcorrect;

distcorrectpluswrong = distcorrect + distwrong;
distcorrectminuswrong = distcorrect - distwrong;

mu = distcorrectminuswrong ./ distcorrectpluswrong;
regTerm = 0;
normTerm = 0;
% if exist('A','var')
if regularization
%         regTerm = regularization * log(det(A'*A)+eps);
%         regTerm = regularization * log(sqrt(diag(A))'*sqrt(diag(A)));
%     if length(relIdx)==1
%             regTerm = regularization * sum(log(-diag(A).^2));
    for j=1:length(Aidx)      
         regTerm = regTerm + regularization * log(det(A(Aidx{j},:)*A(Aidx{j},:)'));       
    end
end
for j=1:length(Aidx)   
      %A(Aidx{j},:)=A(Aidx{j},:)./sqrt(sum(A(Aidx{j},:).^2));
%       normTerm = normTerm + (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),Aidx{j})))^2;
    normTermTemp(Aidx{j}) =sum(A(Aidx{j},:).^2);
   %  normTerm = normTerm + (1-sum(sum(R(:,:,j))))^2;
end
%sum(ones(k,1)-sum(Rs.^2,2))
%normTerm=normTerm+sum(normTermTemp);
normTerm=normTerm+sum((ones(1,length(Aidx))-normTermTemp).^2);
% normTerm = (1-sum(arrayfun(@(d) sum(A(d,:).*A(d,:)),1:length(relIdx))))^2;
% end
f = sum(mu) - regTerm + normTerm;
% x0 = diag(A);
% g = Grad(@(x0) (1-sum(diag(A).^2))^2, x0);
% test = reshape(g,length(g)/size(trainX,2),size(trainX,2))
% g = Grad(@(x0) sum(log(diag(A).^2)), x0);
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
%     if isempty(relIdx)
%         for k=1:nb_prototypes % update all prototypes        
%             actNW3 = normw(k)^3;
%             idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
%             idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong
% 
%             xdiffw = trainX(idxc,:).*normw(k)^2 - bsxfun(@times,xtw(idxc,k),w(k,:));
%             xdiffw(isnan(trainX(idxc,:))) = 0;
%             gwj = sum(bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normx(idxc)*actNW3)));
% 
%             xdiffw = trainX(idxw,:).*normw(k)^2 - bsxfun(@times,xtw(idxw,k),w(k,:));
%             xdiffw(isnan(trainX(idxw,:))) = 0;
%             gwk = sum(bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normx(idxw)*actNW3)));
% 
%             G(k,2:end) = gwj+gwk;
%         end
%     else % using relevances
    if Lprototypes,
        for k=1:nb_prototypes % update all prototypes        
            actNW3 = normwA(k)^3;
            idxc = (k == pidxcorrect);  % Js: idxs where actual prototype is nearest correct
            idxw = (k == pidxwrong);    % Ks: idxs where actual prototype is nearest wrong
%             if length(relIdx)>1
%                 xdiffw = useX(idxc,:)*A'*A .* normwA(k)^2 - bsxfun(@times,xAw(idxc,k) , w(k,:)*A'*A);
%                 gwj = sum(bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));
%                 
%                 xdiffw = useX(idxw,:)*A'*A .* normwA(k)^2 - bsxfun(@times,xAw(idxw,k) , w(k,:)*A'*A);
%                 gwk = sum(bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));            
%             else
            if classwise, useAidx = Aidx{c_A==c_w(k)};else useAidx = Aidx{k};end
                R=(A(useAidx,:));
                xdiffw = bsxfun(@times,R.^2,trainX(idxc,:).*normwA(k)^2 - bsxfun(@times,xAw(idxc,k),w(k,:)));
                xdiffw(isnan(trainX(idxc,:))) = 0;
                %xdiffw =useX(idxc,:)*(A(:,:,useAidx)'*A(:,:,useAidx)) .* normwA(k)^2 - bsxfun(@times,xAw(idxc,k) , w(k,:)*(A(:,:,useAidx)'*A(:,:,useAidx)));
                %xdiffw =useX(idxc,:)*R .* normwA(k)^2 - bsxfun(@times,xAw(idxc,k) , w(k,:)*R);
                gwj = sum(bsxfun(@times,(mudJ(idxc)' .* dgwJ(idxc)') , bsxfun(@rdivide,xdiffw,normxA(idxc,k)*actNW3)));
                
                xdiffw = bsxfun(@times,R.^2,trainX(idxw,:).*normwA(k)^2 - bsxfun(@times,xAw(idxw,k),w(k,:)));
                xdiffw(isnan(trainX(idxw,:))) = 0;
                gwk = sum(bsxfun(@times,(mudK(idxw)' .* dgwK(idxw)'), bsxfun(@rdivide,xdiffw,normxA(idxw,k)*actNW3)));
                %xdiffw = useX(idxw,:)*diag(A(:,:,useAidx)'*A(:,:,useAidx)) .* normwA(k)^2 - bsxfun(@times,xAw(idxw,k) , w(k,:)*diag(A(:,:,useAidx)'*A(:,:,useAidx)));
                %xdiffw = useX(idxw,:)*R .* normwA(k)^2 - bsxfun(@times,xAw(idxw,k) , w(k,:)*R);
                %xdiffw = bsxfun(@times,diag(A(:,useAidx))'.^2,trainX(idxw,:).*normwA(k)^2 - bsxfun(@times,xAw(idxw,k),w(k,:)));
                %gwk = sum(bsxfun(@times,(mudK(idxw)' .* dgwK(idxw)') , bsxfun(@rdivide,xdiffw,normxA(idxw,k)*actNW3)));
                %end
                G(k,2:end) = gwj+gwk;
         end
     end
        % update relevance vector
    if Lrelevances,
        f3 = zeros(length(relIdx),size(trainX,2));normF3 = zeros(length(relIdx),size(trainX,2));
        ga = zeros(size(A));
%         if length(relIdx)>1
% ga = zeros(size(A));
% for d=1:size(A,1)
            % dcosaJdA = bsxfun(@rdivide,bsxfun(@times,trainX,w(pidxcorrect,:)*A(d,:)') + bsxfun(@times,w(pidxcorrect,:),useX*A(d,:)'),normxA.*normwA(pidxcorrect)) - ...
            %            bsxfun(@times,arrayfun(@(j) xAw(j,pidxcorrect(j)),1:length(pidxcorrect))' , ...
            %            (bsxfun(@rdivide,bsxfun(@times,trainX,useX*A(d,:)'),normxA.^3.*normwA(pidxcorrect)) + ...
            %             bsxfun(@rdivide,bsxfun(@times,w(pidxcorrect,:),w(pidxcorrect,:)*A(d,:)'),normxA.*normwA(pidxcorrect).^3) ));
            % dcosaKdA = bsxfun(@rdivide,bsxfun(@times,trainX,w(pidxwrong,:)*A(d,:)') + bsxfun(@times,w(pidxwrong,:),useX*A(d,:)'),normxA.*normwA(pidxwrong)) - ...
            %            bsxfun(@times,arrayfun(@(j) xAw(j,pidxwrong(j)),1:length(pidxwrong))' , ...
            %            (bsxfun(@rdivide,bsxfun(@times,trainX,useX*A(d,:)'),normxA.^3.*normwA(pidxwrong)) + ...
            %             bsxfun(@rdivide,bsxfun(@times,w(pidxwrong,:),w(pidxwrong,:)*A(d,:)'),normxA.*normwA(pidxwrong).^3) ));
% ga(d,:) = nansum( bsxfun(@times,(mudJ.*dgwJ)',dcosaJdA) + bsxfun(@times,(mudK.*dgwK)',dcosaKdA) );
% end
%             normF3 = -4*(1-sum(diag(A'*A))).*A;
%             if regularization,regTerm = regularization.*2.*(pinv(A))';end
%         else
        for d=1:size(A,1)
%                 dcosaJdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX,normwA(pidxcorrect).^2),bsxfun(@times,w(pidxcorrect,:),normxA.^2)) - ...
%                            bsxfun(@times,arrayfun(@(c) xAw(c,pidxcorrect(c)),1:length(pidxcorrect))',bsxfun(@times,trainX.^2,normwA(pidxcorrect).^2)+bsxfun(@times,w(pidxcorrect,:).^2,normxA.^2)), ...
%                            normxA.^3.*normwA(pidxcorrect).^3),diag(A(d,:))');
% 
%                 dcosaKdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX,normwA(pidxwrong).^2),bsxfun(@times,w(pidxwrong,:),normxA.^2)) - ...
%                            bsxfun(@times,arrayfun(@(c) xAw(c,pidxwrong(c)),1:length(pidxwrong))',bsxfun(@times,trainX.^2,normwA(pidxwrong).^2)+bsxfun(@times,w(pidxwrong,:).^2,normxA.^2)), ...
%                            normxA.^3.*normwA(pidxwrong).^3),diag(A(d,:))');
          actIdx = find(arrayfun(@(x) length(find(d==Aidx{x})),1:length(Aidx)));
         % prots = find(c_A(actIdx)==c_w);
          %for pidx = 1:length(prots)
%                 actwJ = find(pidxcorrect==prots(pidx));
%                 actwK = find(pidxwrong==prots(pidx));
                dcosaJdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX',normwA(pidxcorrect).^2),bsxfun(@times,w(pidxcorrect,:),normxA(:,d).^2)') - ...
                           bsxfun(@times,arrayfun(@(c) xAw(c,pidxcorrect(c)),1:length(pidxcorrect)),bsxfun(@times,trainX'.^2,normwA(pidxcorrect).^2)+bsxfun(@times,w(pidxcorrect,:).^2,normxA(:,d).^2)'), ...
                           normxA(:,d)'.^3.*normwA(pidxcorrect).^3),(A(d,:)'))';

                dcosaKdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX',normwA(pidxwrong).^2),bsxfun(@times,w(pidxwrong,:),normxA(:,d).^2)') - ...
                           bsxfun(@times,arrayfun(@(c) xAw(c,pidxwrong(c)),1:length(pidxwrong)),    bsxfun(@times,trainX'.^2,normwA(pidxwrong).^2) + bsxfun(@times,w(pidxwrong,:).^2,normxA(:,d).^2)'), ...
                           normxA(:,d)'.^3.*normwA(pidxwrong).^3),(A(d,:)'))';
%                 dcosaJdA =  bsxfun(@rdivide, bsxfun(@times,trainX(actwJ,:),w(prots(pidx),:)*A(d,:)') + ...
%                             bsxfun(@times,w(prots(pidx),:),useX(actwJ,:)*A(d,:)'), normxA(actwJ,prots(pidx)).*normwA(prots(pidx)) ) - ...
%                             bsxfun(@times, arrayfun(@(j) xAw(actwJ(j),prots(pidx)),1:length(actwJ))' , (...
%                             bsxfun(@rdivide, bsxfun(@times,trainX(actwJ,:),useX(actwJ,:)*A(d,:)'), normxA(actwJ,prots(pidx)).^3.*normwA(prots(pidx))) + ...
%                             bsxfun(@rdivide, bsxfun(@times,w(prots(pidx),:),w(prots(pidx),:)*A(d,:)'),normxA(actwJ,prots(pidx)).*normwA(prots(pidx)).^3)  ));
%             
%                 dcosaKdA =  bsxfun(@rdivide, bsxfun(@times,trainX(actwK,:),w(prots(pidx),:)*A(d,:)') + ...
%                             bsxfun(@times,w(prots(pidx),:),useX(actwK,:)*A(d,:)'), normxA(actwK,prots(pidx)).*normwA(prots(pidx)) ) - ...
%                             bsxfun(@times, arrayfun(@(j) xAw(actwK(j),prots(pidx)),1:length(actwK))' , (...
%                             bsxfun(@rdivide, bsxfun(@times,trainX(actwK,:),useX(actwK,:)*A(d,:)'), normxA(actwK,prots(pidx)).^3.*normwA(prots(pidx))) + ...
%                             bsxfun(@rdivide, bsxfun(@times,w(prots(pidx),:),w(prots(pidx),:)*A(d,:)'),normxA(actwK,prots(pidx)).*normwA(prots(pidx)).^3)  ));                
            %   ga(d,:) = ga(d,:) + nansum( bsxfun(@times,(mudJ.*dgwJ)',dcosaJdA) ) + nansum( bsxfun(@times,(mudK.*dgwK)',dcosaKdA) );
         % end
        end
%        regTerm = zeros(size(A));
%         normF3 = zeros(size(A));
        for j=1:length(Aidx) 
            %normF3(Aidx{j},:)=-4*(1-sum( sum(R(:,j)))).*A(:,:,j);
             normF3(Aidx{j},:)=-4*(1-sum(A(Aidx{j},:).^2)).*A(Aidx{j},:);
        end
        
        if regularization,
            for j=1:length(Aidx)
                %regTerm(Aidx{j},:) = regularization.*2.*(pinv(A(Aidx{j},:)))';
                f3(j,:)=regularization*(2./(A(d,:).^2).*A(d,:))';
            end
        end
        ga = nansum(bsxfun(@times,(mudJ.*dgwJ)',dcosaJdA) + bsxfun(@times,(mudK.*dgwK)',dcosaKdA)) -f3;
%         end
        G(relIdx,2:end) = ga + normF3 - regTerm;% - sqrt(sum(ga.^2)).*diag(A)';
    end
%     end
end
end
% ca2d = @(cosa,beta) 1/(exp(beta)-exp(-beta))*(exp(-beta*cosa)-exp(-beta));
% cosa = @(x,y) (x(find(~isnan(x)))*y(find(~isnan(x)))')/(norm(x(find(~isnan(x))))*norm(y(find(~isnan(y)))));
% cas = nan(size(trainX,1),size(variables,1));
% for i=1:size(trainX,1)
%     for j=1:size(variables,1)
%         cas(i,j) = ca2d(cosa(trainX(i,:),variables(j,2:end)),beta);
%     end
% end

% x=[0,1];y=[1,1];R = [0.5,0.5];
% y = [-1,-2];R = [0.7,0.3];
% sum(arrayfun(@(i) x(i)*y(i),1:2))/(sqrt(sum(x.^2))*sqrt(sum(y.^2)));acosd(ans)
% sum(arrayfun(@(i) x(i)*y(i)*R(i),1:2))/(sqrt(sum(x.^2.*R))*sqrt(sum(y.^2.*R)));acosd(ans)
% x.*R*y'/(sqrt(x.*R*x')*sqrt(y.*R*y'))
% x*diag(R)*y'/( sqrt(x*diag(R)*x')*sqrt(y*diag(R)*y') )
% x*diag(R)*y'/( sqrt(x.*R*x'*y.*R*y') )