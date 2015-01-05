function [mp1, mp2] =  onLineMatches( Gray1, Gray2, dzy1, dzy2, Disp )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

I1 = imfilter(Gray1, fspecial('gaussian', 3));
I2 = imfilter(Gray2, fspecial('gaussian', 3));
% I1=Gray1; I2=Gray2;
I1 = abs(imfilter(I1, fspecial('sobel')'));
I2 = abs(imfilter(I2, fspecial('sobel')'));

allerr=[]; mp1=[]; mp2=[];
for y=1:size(I1, 1)
    % show one line result
    %     subplot(2, 2, 1);
    %     imshow(Gray1);
    %     hold on, plot([1, size(I1, 2)], [y, y]), hold off;
    %     subplot(2, 2, 2);
    %     imshow(Gray2);
    %     hold on, plot([1, size(I2, 2)], [y, y]), hold off;
    %     subplot(2, 2, 3);
    %     plot(I1(y, :));
    %     subplot(2, 2, 4);
    %     plot(I2(y, :));
    
    % extract features
    edge1 = I1(y, :)>0.2;
    pt1 = [find(edge1); repmat(y, 1, nnz(edge1))];
    
    edge2 = I2(y, :)>0.2;
    pt2 = [find(edge2); repmat(y, 1, nnz(edge2))];
    
    % compute descriptors
    %     [feat1, validPt1] = extractFeatures(Gray1, pt1', 'Method', 'Block', 'BlockSize', 3);
    %     [feat2, validPt2] = extractFeatures(Gray2, pt2', 'Method', 'Block', 'BlockSize', 3);
    feat1 = dzy1.descs( (pt1(2, :)-1)*dzy1.w+pt1(1, :), :);
    feat2 = dzy2.descs( (pt2(2, :)-1)*dzy2.w+pt2(1, :), :);
    indexPairs = matchFeatures(feat1,feat2, 'Metric', 'normxcorr'); %, 'MatchThreshold', 2.0, 'MaxRatio', 0.5);
    matchedPt1 = pt1(:, indexPairs(:, 1))';
    matchedPt2 = pt2(:, indexPairs(:, 2))';
    
    % filter matches invalid out
    isValid = matchedPt1(:, 1) - matchedPt2(:, 1) >=0;
    matchedPt1 = matchedPt1(isValid, :);
    matchedPt2 = matchedPt2(isValid, :);
    assert( nnz( matchedPt1(:, 1) - matchedPt2(:, 1) < 0) ==0);
    
    isBorder = matchedPt1(:,1)<3 | size(I1, 2)-matchedPt1(:,1) < 3 ...
    | matchedPt2(:,1)<3 | size(I2, 2)-matchedPt2(:,1) < 3 ...
    | matchedPt1(:,2)<3 | size(I1, 1)-matchedPt1(:,2) < 3 ...
    | matchedPt2(:,1)<3 | size(I2, 2)-matchedPt2(:,1) < 3;

    matchedPt1 = matchedPt1(~isBorder, :);
    matchedPt2 = matchedPt2(~isBorder, :);
    
    % evaluation by GT
    ERR=[Disp(sub2ind(size(Disp), matchedPt1(:, 2), matchedPt1(:, 1))), matchedPt1(:, 1)-matchedPt2(:, 1)];
    
    % add to matched points set
    mp1 = [mp1; matchedPt1];
    mp2 = [mp2; matchedPt2];
    errThisline = abs(ERR(:, 1)-ERR(:, 2));
    errThisline(ERR(:, 1)==0, :) = 0;
    allerr = [allerr; errThisline];
    
    % imshow(Gray1)
    % hold on
    % plot(matchedPt1(:, 2), matchedPt1(:, 1), 'r.')
    % imshow(Gray2)
    % imshow(Gray1)
    % hold on
    % plot(matchedPt1(:, 2), matchedPt1(:, 1), 'r.')
    % figure, imshow(Gray2)
    % hold on
    % plot(matchedPt2(:, 2), matchedPt2(:, 1), 'r.')
    
    %  figure, showMatchedFeatures(Gray1', Gray2', fliplr(matchedPt1), fliplr(matchedPt2), 'montage');
end
% show points precision
imshow(Gray1);
hold on
plot(mp1(round(allerr)<3, 1), mp1(round(allerr)<3, 2), 'g.');
plot(mp1(round(allerr)>=3, 1), mp1(round(allerr)>=3, 2), 'r.');

end

