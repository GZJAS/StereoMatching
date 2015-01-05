%%% input: rectified left and right images, dmax, initial points
%%% output: disparity map
addpath D:\Documents\MATLAB\mdaisy-v1.0

GrayTx = imread('D:\Documents\GitHub\CrossScaleStereo\x64\Release\test03_txhd_L.png');
if 1
    %% read images
    wtDir = 'D:\Documents\GitHub\Rectification\rectified\';
    ORIGINAL_DATASET = 'test03';
    DATASET =  [ORIGINAL_DATASET '_0911'];
    Gray1 = imread([wtDir DATASET '_L.jpg']);
    Gray1=Gray1(:,:,1);
    Gray2 = imread([wtDir DATASET '_R.jpg']);
    Gray2=Gray2(:,:,1);
    Disp = double(imread('D:\Documents\Qualcomm\2014spring\Yang\test03_GT.png'));
    
    % daisy descriptor
    dzy1 = compute_daisy(Gray1);
    dzy2 = compute_daisy(Gray2);
    % extract and match on-line features
    [mp1, mp2] = onLineMatches(Gray1, Gray2, dzy1, dzy2, Disp);
end

if 1
    %% clustering
    ids = clusterSeeds(Gray1, mp1, mp2, 0.5, 0.25); K=max(ids);
    [h, w] = size(Gray1);
    sparse_disp = mp1(:, 1)-mp2(:, 1);
    imshow(cat(3, Gray1, Gray1, Gray1))
    hold on
    scatter(mp1(:,1), mp1(:,2), 36, ids, 'filled');
    colorMap = hsv(K);
    colormap(colorMap)
    caxis([1 K])
    %
    % %% 3 d plot
    % % figure
    % % scatter3(Data(:, 1), Data(:, 2), Data(:, 3), 10, ids, 'filled');
    % % colormap(hsv(K))
    % % set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
end

if 1
    %% refine cluster
    for ki =1:K
        inCluster = ids==ki;
        
        if (nnz(inCluster)<5)
            continue;
        end
        
        % pruning
        clusterIdx = find(inCluster);
        mp1InClus = mp1(inCluster,:); mp2InClus = mp2(inCluster,:);
        
        if nnz(unique(mp1InClus(:,1)))==1 || nnz(unique(mp1InClus(:,2)))==1 ...
                || nnz(unique(mp2InClus(:,1)))==1 || nnz(unique(mp2InClus(:,2)))==1
            continue;
        end
        tform = estimateGeometricTransform(mp1InClus, mp2InClus, 'projective');
        mp2InClus_prime = transformPointsForward(tform, mp1InClus);
        isOutlier = sum( (mp2InClus-mp2InClus_prime).^2, 2 ) > 0.25;
        ids(clusterIdx(isOutlier)) = 0;
        
        % merge
        inCluster = ids==ki;
        if (nnz(inCluster)<5)
            continue;
        end
        mp1InClus = mp1(inCluster,:); mp2InClus = mp2(inCluster,:);
        tform = estimateGeometricTransform(mp1InClus, mp2InClus, 'projective');
        mp2_prime = transformPointsForward(tform, mp1);
        allScore = sum( (mp2-mp2_prime).^2, 2 );
        fitScore = mean(allScore(inCluster,:));
        if fitScore < 0.25
            ids(allScore < max(allScore(inCluster,:))) = ki;
        else
            ids(inCluster) = 0;
        end
    end
    
    imshow(cat(3, Gray1, Gray1, Gray1));
    hold on
    scatter(mp1(ids~=0,1), mp1(ids~=0,2), 36, ids(ids~=0), 'filled');
    
    colorMap = hsv(K);
    colormap(colorMap)
    caxis([1 K])
    
end

if 1
    %% superpixelS
    
    Segments = vl_slic(single(GrayTx), 40, 0.01);
    [Gmag, ~] = imgradient(Segments);
    Show = Gray1; Show(Gmag~=0) = 255;
    imshow(cat(3, Show, Gray1, Gray1));
end

if 1
%% fit plane
    EstDisp=zeros(h, w);
    % for eachh cluster
    Pi = []; pi = 1;
    % figure, image(cat(3, Gray1, Gray1, Gray1));
    % axis equal
    % grid on
    % ids2 = ids;
    fitTable = [];
    Show = zeros(h,w); ShowG = zeros(h,w); ShowB = zeros(h,w);
    
    for ki=1:K
        inCluster = ids==ki;
        
        fitTable(:, ki) = inf(length(mp1), 1);
        if (nnz(inCluster)<5)
            continue;
        end
        clusterData = [mp1(inCluster,:) sparse_disp(inCluster,:)];
        %     imshow(cat(3, Gray1, Gray1, Gray1))
        %     hold on
        %     plot(clusterData(:,1), clusterData(:, 2), 'ro');
        
        % bounding box
        ULx=min(clusterData(:,1));
        ULy=min(clusterData(:,2));
        BRx=max(clusterData(:,1));
        BRy=max(clusterData(:,2));
        % degenerate case
        if BRx-ULx < 1 || BRy-ULy<1
            continue;
        end
        %     region=rectangle('Position',[ULx,ULy,BRx-ULx,BRy-ULy], 'EdgeColor', colorMap(ki,:));
        [yInBox, xInBox] = meshgrid(ULy:BRy, ULx:BRx);  % bounding box region
        %     plot3(corner([1:end, 1],1), corner([1:end, 1],2), cornerDisp([1:end,1])*100, 'Color', colorMap(ki,:));
        
        % belonged superpixel
        inSeg = Segments(sub2ind([h,w], clusterData(:,2), clusterData(:,1)));
        inSegSet = unique(inSeg);
        isInSet = false(h,w);
        for i=1:length(inSegSet)
            isInSet = isInSet | Segments==inSegSet(i);
        end
        %     [Gmag, ~] = imgradient(isInSet);
        
        [yInSuperPx, xInSuperPx] = find(isInSet);  % superpixel region
        
        overlap = isInSet(sub2ind([h, w], yInBox, xInBox));
        xInOverlap = xInBox(overlap); yInOverlap = yInBox(overlap);  % overlap region
        
        xyInR = [xInOverlap(:), yInOverlap(:)];
        Show(sub2ind([h, w], xyInR(:,2), xyInR(:,1))) = colorMap(ki,1);
        ShowG(sub2ind([h, w], xyInR(:,2), xyInR(:,1))) = colorMap(ki,2); ShowB(sub2ind([h, w], xyInR(:,2), xyInR(:,1))) = colorMap(ki,3);
        
        % fit plane
        % try on plane
        A=[clusterData(:,1:2), ones(size(clusterData,1), 1)];
        b=clusterData(:,3);
        Pi(:, pi) = A\b;
        
        %     fitErr = abs(A*Pi(:, pi) - b);
        %     %     sortFitErr = sort(fitErr);
        %     fitScore = rms(fitErr);
        %
        %     %     fitAllErr = abs([mp1, ones(length(mp1), 1)]*Pi(:, pi) - sparse_disp);
        %     %     plot(mp1(fitAllErr<0.5, 1), mp1(fitAllErr<0.5, 2), 'go');
        %
        %     %         ids2(ids2 == ki) = 0;
        %     if fitScore < 0.5
        %         fitAllErr = abs([mp1, ones(length(mp1), 1)]*Pi(:, pi) - sparse_disp);
        %         fitTable(:, ki) = fitAllErr';
        %     else
        %         fitTable(:, ki) = inf(length(mp1), 1);
        %         %continue;
        %     end
        %
        
        % try on homography
        mp1InClus = mp1(inCluster,:); mp2InClus = mp2(inCluster,:);
        [tform, mp1InClusInlier, mp2InClusInlier ]= estimateGeometricTransform(mp1InClus, mp2InClus, 'projective');
        %     xy_warped = transformPointsForward(tform, xyInR);
        
        mp2_prime = transformPointsForward(tform, mp1);
        
        % fit error
        allScore = sum( (mp2-mp2_prime).^2, 2 );
        % TODO:
        % 1. check fit score
        fitScore = mean(allScore(inCluster,:));
        if 0 && fitScore < 0.25
            % 2. identify full region range
            UL = min(mp1(allScore<0.25,:));
            BR = max(mp1(allScore<0.25,:));
            [yInBox, xInBox] = meshgrid(UL(2):BR(2), UL(1):BR(1));
            
            close all
            imshow(Gray1);
            hold on, plot(xInBox, yInBox, 'r+');
            plot(mp1(allScore<fitScore,1), mp1(allScore<fitScore,2), 'g.');
            % 3. warp to correct range
            % 4. (optional) left right shift to cost
        end
        
        %     EstDisp(sub2ind([h,w], xyInR(:, 2), xyInR(:, 1))) = max(curDisp, [xyInR, ones(length(xyInR), 1)]*Pi(:, pi) );
        
        curDisp = EstDisp(sub2ind([h,w], xyInR(:, 2), xyInR(:, 1)));
        xy_warped = transformPointsForward(tform, xyInR);
        EstDisp(sub2ind([h,w], xyInR(:, 2), xyInR(:, 1))) = max(curDisp, xyInR(:,1)-xy_warped(:,1));
        %     EstDisp(isInSet) = max(curDisp, [x, y, ones(length(y), 1)]*Pi(:, pi) );
        
        pi = pi+1;
        %     close all
        %     imshow(cat(3, Gray1, Gray1, Gray1)); hold on;
        
        
    end
end

%% showing result
figure, imshow(cat(3, Show, ShowG, ShowB));

AC=abs(Disp-EstDisp)<=3 & Disp~=0 & EstDisp~=0;
WA=abs(Disp-EstDisp)>3 & Disp~=0 & EstDisp~=0;
UNDEF=EstDisp==0;
figure, imshow(cat(3, double(WA), double(AC), double(UNDEF)));
figure, imshow(uint8(EstDisp./43*256));
imwrite(uint8(EstDisp), 'estDisp.png');

system(['D:\Documents\GitHub\CrossScaleStereo\x64\Release\SSCA.exe' ...
    ' RAWDISP ST WM 0' ...
    ' D:\Documents\GitHub\CrossScaleStereo\x64\Release\test03_txhd_L.png' ...
    ' D:\Documents\GitHub\CrossScaleStereo\x64\Release\test03_txhd_R.png' ...
    ' D:\Documents\GitHub\CrossScaleStereo\x64\Release\disp.png 43 1']);

disp1 = double(imread('D:\Documents\GitHub\CrossScaleStereo\x64\Release\disp.png'));

badPixel = abs(Disp-disp1)>3 & Disp~=0;
totalPixel = Disp~=0;

dmax = 43;
imshow(badPixel)
figure, imshow(uint8(Disp))
figure, imshow(uint8(disp1./dmax.*255));

errRat = nnz(badPixel)/nnz(totalPixel)

return;

% [minVal, minIdx] = min(fitTable, [], 2);
% minIdx(minVal>= 0.5) = 0;
% imshow(cat(3, Gray1, Gray1, Gray1)); hold on;
% scatter(mp1(minIdx~=0,1), mp1(minIdx~=0,2), 36, minIdx(minIdx~=0), 'filled');
% colormap(colorMap)
% caxis([1 K])

% eval fittness
% prune outlier
% further merge or split

%% add value
% EstDisp2 = zeros(h, w);
% for ki=1:K
%     inCluster = minIdx==ki;
%     if (nnz(inCluster)<5)
%         continue;
%     end
%     clusterData = [mp1(inCluster,:) sparse_disp(inCluster,:)];
%
%
%     ULx=min(clusterData(:,1));
%     ULy=min(clusterData(:,2));
%     BRx=max(clusterData(:,1));
%     BRy=max(clusterData(:,2));
%     % degenerate case
%     if BRx-ULx < 1 || BRy-ULy<1
%         continue;
%     end
%     region=rectangle('Position',[ULx,ULy,BRx-ULx,BRy-ULy]);
%     [xInR, yInR] = meshgrid(ULx:BRx, ULy:BRy);
%     xyInR = [xInR(:), yInR(:)];
%
%     % fit plane
%     % try on plane
%     A=[clusterData(:,1:2), ones(size(clusterData,1), 1)];
%     b=clusterData(:,3);
%     Pi(:, pi) = A\b;
%
%     hold on
%     scatter3(clusterData(:,1),clusterData(:,2),clusterData(:,3)*100, 12, colorMap(ki,:));
%     corner = [ULx, ULx, BRx, BRx; ULy, BRy, BRy, ULy]';
%     cornerDisp = [corner, ones(4, 1)]*Pi(:, pi);
%     plot3(corner([1:end, 1],1), corner([1:end, 1],2), cornerDisp([1:end,1])*100, 'Color', colorMap(ki,:));
%
%     fitErr = abs(A*Pi(:, pi) - b);
%     sortFitErr = sort(fitErr);
%     score = mean(sortFitErr(1:round(length(fitErr)*0.8)).^2);
%
%     if score < 0.25
%         curDisp = EstDisp2(sub2ind([h,w], yInR(:), xInR(:)));
%         EstDisp2(sub2ind([h,w], yInR(:), xInR(:))) = max(curDisp, [xyInR, ones(length(xyInR), 1)]*Pi(:, pi) );
%     end
%
%
%
%     pi = pi+1;
% end