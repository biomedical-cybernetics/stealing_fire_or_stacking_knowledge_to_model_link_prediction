function scores = stacking_topological_CH_SPM_SBM_wrapper(A, A_train, edges_train, labels_train, fileprefix, SBM_scores, SBM_scores_train)

[e1,e2] = find(triu(A==0,1));
edges_test = [e1,e2]; clear e1 e2;

% CH
CH_methods = {'RA_L2','CH1_L2','CH2_L2','CH3_L2','RA_L3','CH1_L3','CH2_L3','CH3_L3'};
scores = CHA_linkpred_monopartite(A, CH_methods, 0, 1);
scores_train = CHA_linkpred_monopartite(A_train, CH_methods, 0, 1);
scores = join(array2table(edges_test,'VariableNames',{'node1','node2'}),scores);
scores_train = join(array2table(edges_train,'VariableNames',{'node1','node2'}),scores_train);

% SPM
SPM_scores = SPM_carlo_edited(A);
SPM_scores_train = SPM_carlo_edited(A_train);
scores.SPM = SPM_scores(sub2ind(size(SPM_scores),edges_test(:,1),edges_test(:,2))); clear SPM_scores;
scores_train.SPM = SPM_scores_train(sub2ind(size(SPM_scores_train),edges_train(:,1),edges_train(:,2))); clear SPM_scores_train;

% SBM
scores = join(scores,array2table(SBM_scores,'VariableNames',{'node1','node2','SBM'})); clear SBM_scores;
scores_train = join(scores_train,array2table(SBM_scores_train,'VariableNames',{'node1','node2','SBM'})); clear SBM_scores_train;

fileinput = [fileprefix '_stacking_topological_CH_SPM_SBM_input.mat'];
fileoutput = [fileprefix '_stacking_topological_CH_SPM_SBM_output.mat'];

column_names = scores.Properties.VariableNames(3:end);
scores = table2array(scores(:,3:end));
scores_train = table2array(scores_train(:,3:end));

save(fileinput, 'A', 'A_train', 'edges_train', 'labels_train', 'edges_test', 'column_names', 'scores', 'scores_train')

command = ['python stacking_topological_CH_SPM_SBM_main.py ' fileinput ' ' fileoutput];
system(command);

load(fileoutput, 'scores')

delete(fileinput);
delete(fileoutput);
