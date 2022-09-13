load('../../data/networks_info.mat', 'networks')
N = NaN(length(networks),1);
for i = 1:length(networks)
    load(['../../data/matrices/' networks{i} '.mat'], 'x')
    N(i) = length(x);
end
[~, idx] = sort(N);
networks = networks(idx);

fid = fopen('networks.txt','w');
fprintf(fid,'%s\n', networks{:});
fclose(fid);