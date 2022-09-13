load('../../data/networks_info.mat', 'networks')
linkrem = {'linkrem10','linkrem10_linkrem20noconn'};
iters = 10;
for l = 1:length(linkrem)
    for i = 1:length(networks)
        fileoutput = ['time/' networks{i} '_' linkrem{l} '_SBM_DC_time.mat'];
        if exist(fileoutput,'file')
            continue;
        end
        
        done = NaN(iters,1);
        for j = 1:iters
            fileoutput_j = ['time/' networks{i} '_' num2str(j) '_DC_' linkrem{l} '_time.txt'];
            done(j) = exist(fileoutput_j,'file');
        end
        if any(~done)
            % fprintf('%d/%d: %s ... scores not available!\n', i, length(networks), networks{i});
            continue;
        end
        
        fprintf('%d/%d: %s\n', i, length(networks), networks{i});
        time = NaN(iters,1);
        for j = 1:iters
            fileoutput_j = ['time/' networks{i} '_' num2str(j) '_DC_' linkrem{l} '_time.txt'];
            time(j) = dlmread(fileoutput_j);
        end
        save(fileoutput, 'time')
        clear time
    end
end