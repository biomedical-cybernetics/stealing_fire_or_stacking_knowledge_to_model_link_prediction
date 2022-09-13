% Matlab codes for structural perturbation link prediction method.
% Journal reference:"Toward link predictability of complex networks",
% Proceedings of the National Academy of Sciences, 201424644 (2015).
% by Linyuan L¨¹, Liming Pan, Tao Zhou, Yi-Cheng Zhang and H. Eugene Stanley.
% http://www.pnas.org/content/early/2015/02/05/1424644112.
% Coded by Liming Pan.

function probMatrix=SPM_carlo_edited(AdjTraining) %,fname)
% structuralConsistency(fnameTraining,fname) returns the structural
% consistency of the training network compared to the full network.
% Inputs:  fnameTraining, the file name of the Training network
%          fname, the file name of the full network
% Output:  pre, the prediction accuracy measured by precision
%          AUC, the prediction accuracy measure by AUC
% the network adjacency list is stored in the following format for default:
% The node number starts at 1. Undirected links are stored twice.
% For example, network of two nodes connected by a single link is stored as
% 1 2 1
% 2 1 1

%Load the training network and the full network
%AdjTraining=spconvert(load(fnameTraining));
%AdjTraining=spconvert(fnameTraining);
%Adj=spconvert(load(fname));
%AdjProb=Adj-AdjTraining;
%probeSize=nnz(AdjTraining)/2;
N=length(AdjTraining);
probMatrix=zeros(N,N);

%Set the size of perturbations and number of independent perturbations
pertuSize=ceil(0.1*(length(find(AdjTraining==1)))/2);
perturbations=10;
%eig_list=zeros(1:N,samples);
for pertus=1:perturbations
    AdjUnpertu=AdjTraining;
    index=find(tril(AdjTraining));
    [i,j]=ind2sub(size(tril(AdjTraining)),index);
    for pp=1:pertuSize
        rand_num=ceil(length(i)*rand(1));
        select_ID1=i(rand_num);
        select_ID2=j(rand_num);
        i(rand_num)=[];
        j(rand_num)=[];
        AdjUnpertu(select_ID1,select_ID2)=0;
        AdjUnpertu(select_ID2,select_ID1)=0;
    end
    AdjUnpertu=full(AdjUnpertu);
    probMatrix=probMatrix+perturbation(AdjUnpertu,AdjTraining);
end

%calculate the precision
% index=find(tril(~AdjTraining,-1));
% [row,col]=ind2sub(size(tril(AdjTraining,-1)),index);
% weight=probMatrix(index);
% finale=[row, col, weight];

%[x,y]=sort(weight);
% pre=0;
% for j=(length(y)-probeSize+1):length(y)
%     if Adj(row(y(j)),col(y(j)))==1
%         pre=pre+1;
%     end
% end
% pre=pre./probeSize;

% %calculate the AUC score
% 
% %scores for the probe set links 
% index1=find(tril(AdjProb,-1));
% weight1=probMatrix(index1);
% 
% %scores for non-exist links
% index2=find(tril(~Adj,-1));
% weight2=probMatrix(index2);
% labels=[];
% scores=[];
% labels(1:length(weight1))=1;
% labels(end+1:end+length(weight2))=0;
% scores(1:length(weight1))=weight1;
% scores(end+1:end+length(weight2))=weight2;
% [X,Y,T,AUC] = perfcurve(labels,scores,1);
% return;
% 


%%%%%%%%%%%%%%%%%%%%%%
%% Support Functions %
%%%%%%%%%%%%%%%%%%%%%%

% Matlab codes for computing link predictability of complex networks.
% Journal reference:"Toward link predictability of complex networks", 
% Proceedings of the National Academy of Sciences, 201424644 (2015).
% by Linyuan L¨¹, Liming Pan, Tao Zhou, Yi-Cheng Zhang and H. Eugene Stanley.
% http://www.pnas.org/content/early/2015/02/05/1424644112.
% Coded by Liming Pan.

function AdjAnneal=perturbation(AdjTraining,Adj)
% perturbation(AdjTraining,Adj) returns the perturbated matrix of the original 
% adjaceny matrix.
% Inputs:  AdjTraining, the unperturbated network,
%          Adj, The unperturbated network plus the perturbations.
% Outputs: AdjAnneal, the perturbated matrix of AdjTraining.

% eigen decomposition of AdjTraining
N=length(Adj);
AdjTraining=full(AdjTraining);
[v,w]=eig(double(AdjTraining));
eigenValues=diag(w);

% find "correct" eigenvectors for perturbation of degenerate eigenvalues 
degenSign=zeros(N,1);

%v2 and w2 are the "correct" eigenvectors and eigenvalues
v2=v;
w2=eigenValues;
AdjPertu=Adj-AdjTraining;
for l=1:N
    if degenSign(l)==0
        tempEigen=find(abs((eigenValues-eigenValues(l)))<10e-12);
        if length(tempEigen)>1
            vRedun=v(1:end,tempEigen);
            m=vRedun'*AdjPertu*vRedun;
            m=(m+m')./2;
            m=full(m);
            [v_r,w_r]=eig(m);
            vRedun=vRedun*v_r;
            % renormalized the  new eigenvectors
            for o=1:length(m)
                vRedun(1:end,o)=vRedun(1:end,o)./norm(vRedun(1:end,o));
            end
            v2(1:end,tempEigen)=vRedun;
            w2(tempEigen)=eigenValues(tempEigen)+diag(w_r);
            degenSign(tempEigen)=1;
        end
    end
end

% pertubate the adjacency matrix AdjTraining
AdjAnneal=v2*diag(diag(v2'*Adj*v2))*v2';
return;













