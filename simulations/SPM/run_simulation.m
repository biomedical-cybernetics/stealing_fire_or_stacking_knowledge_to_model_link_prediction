load('../../data/networks_info.mat', 'networks')

measures_names = {'prec','auc_prec','auc_pr','auc_roc','auc_mroc','ndcg','mcc'};

for i = 1:length(networks)
    
    temp = ['TEMP/' networks{i} '_linkrem10_SPM_linkpred.TEMP'];
    outfile = ['results/' networks{i} '_linkrem10_SPM_linkpred.mat'];
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
        scores = SPM_carlo_edited(matrices{j});
        
        % evaluation
        [e1,e2] = find(triu(matrices{j}==0,1));
        labels = x(sub2ind(size(x),e1,e2));
        scores = scores(sub2ind(size(x),e1,e2));
        measures{j} = prediction_evaluation(scores, labels);
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
