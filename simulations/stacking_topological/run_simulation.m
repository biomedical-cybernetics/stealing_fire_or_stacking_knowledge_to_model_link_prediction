load('../../data/networks_info.mat', 'networks')

measures_names = {'prec','auc_prec','auc_pr','auc_roc','auc_mroc','ndcg','mcc'};

for i = 1:length(networks)
    
    temp = ['TEMP/' networks{i} '_linkrem10_stacking_topological_linkpred.TEMP'];
    outfile = ['results/' networks{i} '_linkrem10_stacking_topological_linkpred.mat'];
    if exist(temp,'file') || exist(outfile,'file')
        continue;
    end
    fclose(fopen(temp, 'w'));
    
    load(['../../data/matrices/' networks{i} '.mat'], 'x')
    load(['../../data/sparsified_matrices/' networks{i} '_linkrem10.mat'], 'matrices')
    matrices_train = load(['../../data/sparsified_matrices/' networks{i} '_linkrem10_linkrem20noconn.mat'], 'matrices');
    matrices_train = matrices_train.matrices;
    load(['../../data/stacking_training_set/' networks{i} '_linkrem10_linkrem20noconn_stacking_training_set.mat'], 'edges_train', 'labels_train')
    time = tic;
    fprintf('%d/%d: (N=%d) ... ', i, length(networks), length(x));

    measures = cell(length(matrices),1);
    parfor j = 1:length(matrices)
        % link prediction
        fileprefix = sprintf('TEMP/%d_%d', i, j);
        scores = stacking_topological_wrapper(matrices{j}, matrices_train{j}, edges_train{j}, labels_train{j}, fileprefix);
        
        % evaluation
        e1 = scores(:,1); e2 = scores(:,2);
        labels = x(sub2ind(size(x),e1,e2));
        scores = scores(:,3);
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
