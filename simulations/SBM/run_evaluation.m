load('../../data/networks_info.mat', 'networks')

measures_names = {'prec','auc_prec','auc_pr','auc_roc','auc_mroc','ndcg','mcc'};

for i = 1:length(networks)
    
    scoresfile = ['scores/' networks{i} '_linkrem10_SBM_DC_scores.mat']; 
    outfile = ['results/' networks{i} '_linkrem10_SBM_DC_linkpred.mat'];
    if exist(outfile,'file')
        continue;
    end
    
    load(['../../data/matrices/' networks{i} '.mat'], 'x')
    load(['../../data/sparsified_matrices/' networks{i} '_linkrem10.mat'], 'matrices')
    load(scoresfile, 'scores')
    fprintf('%d/%d: %s (N=%d)\n', i, length(networks), networks{i}, length(x));
    
    measures = cell(length(matrices),1);
    for j = 1:length(matrices)
        labels = x(sub2ind(size(x),scores{j}(:,1),scores{j}(:,2)));
        measures{j} = prediction_evaluation(scores{j}(:,3), labels);
    end
    
    for m = 1:length(measures_names)
        eval(sprintf('%s = NaN(length(matrices),1);', measures_names{m}));
        for j = 1:length(matrices)
            eval(sprintf('%s(j) = measures{j}.%s;', measures_names{m}, measures_names{m}));
        end
    end
    
    save(outfile, measures_names{:})
end
