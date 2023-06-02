function [f G]  = angleGRLVQ_optfun(variables,trainX,LabelEqualsPrototype,beta,regularization,Lprototypes,Lrelevances,dim)
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
w   = variables(wIdx,2:end);%     c_w = variables(wIdx,1);
%     if length(relIdx)>1
%         A = variables(relIdx,2:end);
%     else
A = diag(variables(relIdx,2:end));% diag(sqrt(variables(relIdx,2:end)));
%     end
xAw = useX*A'*A*w';
normxA = sqrt(diag(useX*A'*A*useX'));
normwA = sqrt(diag(w*A'*A*w'));
cosa = bsxfun(@rdivide,xAw,normxA*normwA');
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
% if exist('A','var')
if regularization
%         regTerm = regularization * log(det(A'*A)+eps);
%         regTerm = regularization * log(sqrt(diag(A))'*sqrt(diag(A)));
%     if length(relIdx)==1
%             regTerm = regularization * sum(log(-diag(A).^2));
        regTerm = regularization * sum(log(diag(A).^2));
%         else
%             regTerm = regularization * log(det(A*A'));
%         end
end
normTerm = (1-sum(diag(A).^2))^2;
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
            xdiffw = bsxfun(@times,diag(A)'.^2,trainX(idxc,:).*normwA(k)^2 - bsxfun(@times,xAw(idxc,k),w(k,:)));
            xdiffw(isnan(trainX(idxc,:))) = 0;
            gwj = sum(bsxfun(@times,mudJ(idxc)' .* dgwJ(idxc)' , bsxfun(@rdivide,xdiffw,normxA(idxc)*actNW3)));

            xdiffw = bsxfun(@times,diag(A)'.^2,trainX(idxw,:).*normwA(k)^2 - bsxfun(@times,xAw(idxw,k),w(k,:)));
            xdiffw(isnan(trainX(idxw,:))) = 0;
            gwk = sum(bsxfun(@times,mudK(idxw)' .* dgwK(idxw)' , bsxfun(@rdivide,xdiffw,normxA(idxw)*actNW3)));
%             end
            G(k,2:end) = gwj+gwk;
        end
    end
        % update relevance vector
    if Lrelevances,
        f3 = zeros(length(relIdx),size(trainX,2));normF3 = zeros(length(relIdx),size(trainX,2));
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
        dcosaJdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun(@times,bsxfun(@times,trainX,normwA(pidxcorrect).^2),bsxfun(@times,w(pidxcorrect,:),normxA.^2)) - ...
            bsxfun(@times,arrayfun(@(c) xAw(c,pidxcorrect(c)),1:length(pidxcorrect))',bsxfun(@times,trainX.^2,normwA(pidxcorrect).^2)+bsxfun(@times,w(pidxcorrect,:).^2,normxA.^2)), ...
            normxA.^3.*normwA(pidxcorrect).^3),diag(A)');

        dcosaKdA = bsxfun(@times,bsxfun(@rdivide,2.*bsxfun( @times,bsxfun(@times,trainX,normwA(pidxwrong).^2),bsxfun(@times,w(pidxwrong,:),normxA.^2)) - ...
            bsxfun(@times,arrayfun(@(c) xAw(c,pidxwrong(c)),1:length(pidxwrong))',bsxfun(@times,trainX.^2,normwA(pidxwrong).^2)+bsxfun(@times,w(pidxwrong,:).^2,normxA.^2)), ...
            normxA.^3.*normwA(pidxwrong).^3),diag(A)');
        normF3 = -4*(1-sum(diag(A).^2)).*diag(A)';
        if regularization,
%             f3 = diag(pinv(A'))';
%                 f3 =  regularization*(pinv(diag(A')));
                f3 =  regularization*(2./(diag(A).^2).*diag(A))';
%             normF3 = 2.*diag(A)';                    
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