function scores = stacking_topological_wrapper(A, A_train, edges_train, labels_train, fileprefix)

[e1,e2] = find(triu(A==0,1));
edges_test = [e1,e2]; clear e1 e2;

fileinput = [fileprefix '_stacking_topological_input.mat'];
fileoutput = [fileprefix '_stacking_topological_output.mat'];

save(fileinput, 'A', 'A_train', 'edges_train', 'labels_train', 'edges_test')

command = ['python stacking_topological_main.py ' fileinput ' ' fileoutput];
system(command);

load(fileoutput, 'scores')

delete(fileinput);
delete(fileoutput);