function [scores, CHA_info] = CHA_linkpred_monopartite(x, methods, CHA_option, cores)

%%% INPUT %%%
% x - monopartite adjacency matrix of the network (unweighted, undirected and zero-diagonal).
%
% methods - Cell array of strings indicating the CH models to compute,
%   the possible options are: 'RA_L2','CH1_L2','CH2_L2','CH3_L2','RA_L3','CH1_L3','CH2_L3','CH3_L3'.
%   If empty or not given, methods = {'CH2_L2', 'CH3_L2', 'CH2_L3', 'CH3_L3'}.
%
% CHA_option - The possible options are:
%   CHA_option = 0 -> CHA is not computed.
%   CHA_option = 1 -> CHA is computed over the set of CH models indicated in "methods".
%   CHA_option = cell array of strings indicating a subset of the CH models in "methods" -> CHA is computed over this subset.
%   If empty or not given, CHA_option = 1 if length(methods)>1 and CHA_option = 0 if length(methods)==1.
%   The CHA computation is only valid over at least 2 methods.
%
% cores - number of cores to use for parallel computation.
%   Select 1 for serial computation.
%   If empty or not given, the maximum number available is used.
%
%%% OUTPUT %%%
% scores - Table containing CH scores for node pairs of non-observed links.
%   The first two columns indicate the node pairs.
%   If CHA_option is not 0, the third column contains the scores of the CHA method.
%   Following columns contain the scores of each CH method and of the respective CH-SPcorr subranking.
%   Higher scores suggest higher likelihood of connection between the node pairs.
%
% CHA_info - Structure containing information about the CHA method in the fields:
%   methods -> cell array of strings indicating the CH models over which the CHA is computed
%   aupr -> for each CH model, aupr of discrimination between observed and non-observed links
%   selected_method -> string indicating the CH model selected by the CHA method
%   If CHA_option = 0, CHA_info = [].

% CHA method description:
% For each CH model, all the node pairs are assigned a rank-score (from 1 up)
% while ranking them by increasing CH scores and, in case of tie, by increasing CH-SPcorr scores.
% If they are still tied, they get the same rank-score.
% Therefore, the node pair with highest likelihood of connection gets the highest rank-score.
% For each CH model, using the rank-scores, the discrimination between observed and non-observed links is assessed by aupr.
% The CHA method selects the CH model with the highest aupr and provides in output its rank-scores.

% MEX support function:
% The Matlab function requires the MEX function "CH_scores_mex".
% Compile in Windows:
% Go to MATLAB "Add-Ons" and install "MATLAB Support for MinGW-w64 C/C++ Compiler"
% Build the MEX function using the following MATLAB command (change the MinGW path if needed):
% mex C:\ProgramData\MATLAB\SupportPackages\R2020b\3P.instrset\mingw_w64.instrset\lib\gcc\x86_64-w64-mingw32\6.3.0\libgomp.a CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
% Compile in Linux or Apple Mac:
% Build the MEX functions using the following MATLAB commands:
% mex CH_scores_mex.c CFLAGS='$CFLAGS -fopenmp' LDFLAGS='$LDFLAGS -fopenmp'
% It will generate a MEX file with platform-dependent extension,
% .mexw64 for Windows, .mexa64 for Linux, .mexmaci64 for Apple Mac.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% check input
narginchk(1,4)
validateattributes(x, {'numeric'}, {'square','binary'});
x = sparse(x);
if ~issymmetric(x)
    error('The input matrix must be symmetric.')
end
if any(x(speye(size(x))==1))
    error('The input matrix must be zero-diagonal.')
end
if ~exist('methods', 'var') || isempty(methods)
    methods = {'CH2_L2', 'CH3_L2', 'CH2_L3', 'CH3_L3'};
else
    validateattributes(methods, {'cell'}, {});
    if any(~ismember(methods, {'RA_L2','CH1_L2','CH2_L2','CH3_L2','RA_L3','CH1_L3','CH2_L3','CH3_L3'}))
        error('Possible methods: ''RA_L2'',''CH1_L2'',''CH2_L2'',''CH3_L2'',''RA_L3'',''CH1_L3'',''CH2_L3'',''CH3_L3''.');
    end
    if length(methods) > length(unique(methods))
        error('The variable ''methods'' should not contain duplicates.')
    end
end
if ~exist('CHA_option', 'var') || isempty(CHA_option)
    if length(methods)==1
        CHA_option = [];
    else
        CHA_option = methods;
    end
elseif isnumeric(CHA_option)
    validateattributes(CHA_option, {'numeric'}, {'scalar','binary'});
    if CHA_option == 1
        CHA_option = methods;
    else
        CHA_option = [];
    end
else
    validateattributes(CHA_option, {'cell'}, {});
    if any(~ismember(CHA_option, methods))
        error('The variable ''CHA_option'' contains methods not present in ''methods''.');
    end
    if length(CHA_option) > length(unique(CHA_option))
        error('The variable ''CHA_option'' should not contain duplicates.')
    end
end
if length(CHA_option)==1
    error('The CHA computation is only valid over at least 2 methods.')
end
if ~exist('cores', 'var') || isempty(cores)
    cores = Inf;
else
    validateattributes(cores, {'numeric'}, {'scalar','integer','positive'});
end

% compute CH scores
M = length(methods);
L = NaN(M,1);
models = cell(M,1);
for m = 1:M
    temp = strsplit(methods{m},'_L');
    L(m) = str2double(temp{2});
    models{m} = temp{1};
end
L = unique(L);
models_all = {'RA','CH1','CH2','CH3'};
models = find(ismember(models_all,models))-1;
if isinf(cores)
    scores = CH_scores_mex(x, L, models);
else
    scores = CH_scores_mex(x, L, models, cores);
end
S = cell(length(methods),1);
for i1 = 1:length(L)
    for i2 = 1:length(models)
        method = sprintf('%s_L%d', models_all{models(i2)+1}, L(i1));
        m = find(strcmp(method,methods),1);
        if ~isempty(m)
            S{m} = scores{i1,i2};
            scores{i1,i2} = [];
        end
    end
end

% compute SPcorr and rank scores
if cores == 1 || M == 1
    cores = 0;
end
[e1,e2] = find(triu(true(size(x)),1));
scores = zeros(length(e1),M);
scores_SPcorr = zeros(length(e1),M);
scores_rank = zeros(length(e1),M);
labels = x(sub2ind(size(x),e1,e2));
if isempty(CHA_option)
    parfor (m = 1:M, cores)
        [scores(:,m), scores_SPcorr(:,m), scores_rank(:,m)] = compute_SPcorr_and_rank_scores(x, S{m}, e1, e2);
    end
else
    aupr = NaN(1,M);
    CHA_flag = ismember(methods, CHA_option);
    parfor (m = 1:M, cores)
        [scores(:,m), scores_SPcorr(:,m), scores_rank(:,m)] = compute_SPcorr_and_rank_scores(x, S{m}, e1, e2);
        if CHA_flag(m)
            aupr(m) = compute_aupr(scores_rank(:,m), labels);
        end
    end
end

% output preparation
if isempty(CHA_option)
    CHA_info = [];
    colnames = cell(2+2*length(methods),1);
    colnames(1:2) = {'node1','node2'};
    colnames(3:2+length(methods)) = methods;
    colnames(3+length(methods):2+2*length(methods)) = strcat(methods,'_SPcorr');
    scores = array2table([e1 e2 scores scores_SPcorr],'VariableNames',colnames);
else
    [~, m] = max(aupr);
    CHA_info.methods = strcat(methods(CHA_flag),'_SPcorr');
    CHA_info.aupr = array2table(aupr(CHA_flag),'VariableNames',CHA_info.methods);
    CHA_info.selected_method = strcat(methods{m},'_SPcorr');
    colnames = cell(3+2*length(methods),1);
    colnames(1:3) = {'node1','node2','CHA'};
    colnames(4:3+length(methods)) = methods;
    colnames(4+length(methods):3+2*length(methods)) = strcat(methods,'_SPcorr');
    scores = array2table([e1 e2 scores_rank(:,m) scores scores_SPcorr],'VariableNames',colnames);
end
scores = scores(labels==0,:);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [scores, scores_SPcorr, scores_rank] = compute_SPcorr_and_rank_scores(x, S, e1, e2)

s = x .* 1./(1+S);
s = graphallshortestpaths(sparse(s), 'Directed', false);
s = corr(s, s, 'type', 'spearman');
scores = S(sub2ind(size(x),e1,e2));
scores_SPcorr = s(sub2ind(size(x),e1,e2));
[~,~,scores_rank] = unique([scores scores_SPcorr], 'sorted', 'rows');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function aupr = compute_aupr(scores, labels)

%%% INPUT %%%
% scores - numerical scores for the samples
% labels - binary labels indicating the positive and negative samples

%%% OUTPUT %%%
% aupr   - area under precision-recall

validateattributes(scores, {'numeric'}, {'vector','finite'})
N = length(scores);
validateattributes(labels, {'numeric'}, {'vector','binary','numel',N})
N1 = sum(labels==1);
N0 = N - N1;
if N1==0 || N0==0
    error('labels cannot be all ones or all zeros')
end
if isrow(scores); scores = scores'; end
if isrow(labels); labels = labels'; end

[scores,idx] = sort(-scores, 'ascend');
labels = labels(idx);
[~,ut,~] = unique(scores);
ut = [ut(2:end)-1; N];
tp = full(cumsum(labels));
recall = tp ./ sum(labels);
precision = tp ./ (1:N)';

recall = recall(ut);
precision = precision(ut);
if all(recall==1)
    aupr = precision(1);
else
    aupr = trapz(recall,precision) / (1-recall(1));
end
