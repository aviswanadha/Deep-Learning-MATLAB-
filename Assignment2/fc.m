close all;
clear all;
clc;
run './matconvnet/matlab/vl_setupnn'
net = load('imagenet-caffe-alex.mat') ;

disp('Preparing testing data');
folderCat = './Testing/Cat/';
folderDog = './Testing/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
im = imread([folderCat filename]);
im_ = single(im) ;
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im = imresize(im,[227,227]);
for j = 1:3
    im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
end
% run the CNN
res = vl_simplenn(net, im_) ;
layer_number = 18; %'fc7'
featureVector = res(layer_number).x;
% visualize the classification result
scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
figure ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
end

%% Test on Dog
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
im = imread([folderDog filename]);
im_ = single(im) ;
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im = imresize(im,[227,227]);
for j = 1:3
im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
end
% run the CNN
res = vl_simplenn(net, im_) ;
layer_number = 20; %'fc7'
featureVector = res(layer_number).x;
% visualize the classification result
scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
figure ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
end

disp('Preparing training data');
folderCat = './Training/Cat/';
folderDog = './Training/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));
% Train on Cat
for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
im = imread([folderCat filename]);
im_ = single(im) ;
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im = imresize(im,[227,227]);
for j = 1:3
    im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
end
% run the CNN
res = vl_simplenn(net, im_) ;
layer_number = 18; %'fc7'
featureVector = res(layer_number).x;
% visualize the classification result
scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
figure ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
end

%% Train on Dog
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
im = imread([folderDog filename]);
im_ = single(im) ;
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
im = imresize(im,[227,227]);
for j = 1:3
im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
end
% run the CNN
res = vl_simplenn(net, im_) ;
layer_number = 20; %'fc7'
featureVector = res(layer_number).x;
% visualize the classification result
scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
figure ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
net.meta.classes.description{best}, best, bestScore)) ;
end



