load('../../data/networks_info.mat', 'networks')

measures_names = {'prec','auc_prec','auc_pr','auc_roc','auc_mroc','ndcg','mcc'};
CHA_methods = {'CH2_L2', 'CH3_L2', 'CH2_L3', 'CH3_L3'};

for i = 1:length(networks)
    
    temp = ['TEMP/' networks{i} '_linkrem10_CHA_linkpred.TEMP'];
    outfile = ['results/' networks{i} '_linkrem10_CHA_linkpred.mat'];
    if exist(temp,'file') || exist(outfile,'file')
        continue;
    end
    fclose(fopen(temp, 'w'));
    
    load(['../../data/matrices/' networks{i} '.mat'], 'x')
    load(['../../data/sparsified_matrices/' networks{i} '_linkrem10.mat'], 'matrices')
    time = tic;
    fprintf('%d/%d: %s (N=%d) ... ', i, length(networks), networks{i}, length(x));

    measures = cell(length(matrices),1);
    parfor j = 1:length(matrices)
        % link prediction
        scores = CHA_linkpred_monopartite(matrices{j}, CHA_methods, CHA_methods, 1);
        scores = table2array(scores(:,1:3));
        
        % evaluation
        labels = x(sub2ind(size(x),scores(:,1),scores(:,2)));
        measures{j} = prediction_evaluation(scores(:,3), labels);
    end
    
    for m = 1:length(measures_names)
        eval(sprintf('%s = NaN(length(matrices),1);', measures_names{m}));
        for j = 1:length(matrices)
            eval(sprintf('%s(j) = measures{j}.%s;', measures_names{m}, measures_names{m}));
        end
    end
    
    save(outfile, measures_names{:})
    time = round(toc(time));
    fprintf('[%ds]\n', time);
    delete(temp);
end
