load('../../data/networks_info.mat', 'networks')
linkrem = {'linkrem10','linkrem10_linkrem20noconn'};
iters = 10;
for l = 1:length(linkrem)
    for i = 1:length(networks)
        outscores = ['scores/' networks{i} '_' linkrem{l} '_SBM_DC_scores.mat'];
        if exist(outscores,'file')
            continue;
        end
        
        done = NaN(iters,1);
        for j = 1:iters
            outscores_j = ['output/' networks{i} '_' num2str(j) '_DC_' linkrem{l} '_scores.txt'];
            done(j) = exist(outscores_j,'file');
        end
        if any(~done)
            % fprintf('%d/%d: %s ... scores not available!\n', i, length(networks), networks{i});
            continue;
        end
        
        fprintf('%d/%d: %s\n', i, length(networks), networks{i});
        load(['../../data/sparsified_matrices/' networks{i} '_' linkrem{l} '.mat'], 'matrices')
        scores = cell(iters,1);
        for j = 1:iters
            outscores_j = ['output/' networks{i} '_' num2str(j) '_DC_' linkrem{l} '_scores.txt'];
            scores{j} = dlmread(outscores_j);
            scores{j}(:,1:2) = scores{j}(:,1:2) + 1;
            [e1,e2] = find(triu(matrices{j}==0,1));
            check = sortrows([e1,e2],[1 2])==sortrows(scores{j}(:,1:2),[1 2]);
            if any(~check(:))
                error('Mismatch in output missing links')
            end
            scores{j} = sortrows(scores{j}, -3);
        end
        save(outscores, 'scores', '-v7.3')
        clear scores
    end
end