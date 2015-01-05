function [ ids ] = clusterSeeds(Gray1, mp1, mp2, hs, hr)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[h, w] = size(Gray1);
sparse_disp = mp1(:, 1)-mp2(:, 1);
sparse_disp(sparse_disp==0) = 0.01;
Data = [mp1, 1./sparse_disp*min(sparse_disp) ...
    double(Gray1(sub2ind([h, w], mp1(:, 2), mp1(:, 1)))) ...
    ];
[npt, ~] = size(Data);
T = [[diag(std(Data)); mean(Data)], [zeros(4,1); 1]];
DataNorm = [Data, ones(npt, 1)]/T;
DataNorm(:, 3) = DataNorm(:, 3)*2;
DataNorm(:, 5)=[];
% DataNorm = DataNorm(:, [1,2,4,3]);
eps = 1/16;
[ids, centers] = MeanShift(DataNorm', [hs, hr], eps);

% K=max(ids);
% ids = kmeans(DataNorm, K);

end

