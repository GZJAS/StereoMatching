function [index, centers] = MeanShift( dimg, h, eps )
%hw3-2 h=0 means use kNN adpative bandwidth; length(h)=2 use joint

[nCh, nPx] = size(dimg);
isJoint = length(h)==2;
isUseKNN = h == 0;

if isJoint
    hs = h(1); hr = h(2); % h = [hs, hr]
    h_sqr = h.^2;
    hs_sqr = h_sqr(1); hr_sqr = h_sqr(2);
    weis = hr/(hr+hs); weir = hs/(hr+hs);
elseif isUseKNN
    k = round(0.005*nPx); %0.0011
else
    h_sqr = h^2;
end

eps_sqr = eps^2;
index = zeros(nPx, 1);
centers = zeros(nPx, nCh);  kcount = 1;

for pi=1:nPx % For each point
%         if ~mod(pi, 500)
%             fprintf('%d ', pi);
%             if ~mod(pi, 5000)
%                 fprintf('\n');
%             end
%         end
    
    if index(pi) % has not in cluster yet
        continue;
    end
    
    yj = zeros(nCh, 1); yjp1 = dimg(:, pi);
    isMerged = 0; index(pi) = kcount;
    while sum((yj - yjp1).^2) >= eps_sqr && ~isMerged % untill converge or merged
        yj = yjp1;
        
        % calculate g(x) = -K(x), gradient of kernel
        if isJoint
            dif_sqr = (dimg - repmat(yj, 1, nPx)).^2;
            ecu_dis_sqr_s = sum(dif_sqr(1:2, :));
            ecu_dis_sqr_r = sum(dif_sqr(3:end, :));
            
            gx_spatial = -ecu_dis_sqr_s ./ hs_sqr;
            gx_range = -ecu_dis_sqr_r ./ hr_sqr;            
            
            index(ecu_dis_sqr_s < hs_sqr & ecu_dis_sqr_r < hr_sqr) = kcount; % DP
            gx = exp(gx_spatial + gx_range); 
        else
            ecu_dis_sqr = sum((dimg - repmat(yj, 1, nPx)).^2);
            
            if isUseKNN
                sorted = sort(ecu_dis_sqr);
                h_sqr = max(sorted(k), 1);
            end
            index(ecu_dis_sqr < h_sqr) = kcount; % DP
            gx = exp(-ecu_dis_sqr/h_sqr);
        end
        
        % mean shift vector
        numer = sum( dimg .* repmat(gx, nCh, 1), 2);
        denom = sum(gx);     
        yjp1 = numer./denom;
        
        if kcount > 1 % merge to previous cluster
            % find nearst cluster
            dif_sqr = (centers(1:kcount-1, :) - repmat(yjp1', kcount-1, 1)).^2;
            
            if isJoint
                ecu_dis_s = sqrt(sum(dif_sqr(:, 1:2), 2));
                ecu_dis_r = sqrt(sum(dif_sqr(:, 3:end), 2));
                dis = weis * ecu_dis_s + weir * ecu_dis_r;
                [~, minInd] = min(dis);
                isMerged = ecu_dis_s(minInd) < hs && ecu_dis_r(minInd) < hr;
            else
                [minVal, minInd] = min(sum(dif_sqr, 2));
                isMerged =  minVal < h_sqr;
            end

            if  isMerged % if near enough
                index(index == kcount) = minInd;
            end
        end
    end
    
    % new cluster generated
    if ~isMerged
        centers(kcount, :) = yjp1;
        kcount = kcount +1;
    end
    
end

% return
if isJoint
    centers = centers(1:kcount-1, 1:3);
else
    centers = centers(1:kcount-1, : );
end

end
