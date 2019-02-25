close all;
clear all;
clc;
run './matconvnet/matlab/vl_setupnn'
Lab6c;
% load the trained CNN
load('.\matconvnet\data\dogcat-simplenn\net-epoch-4.mat') ;
net.layers{end}.type = 'softmax';
imdb = load('.\matconvnet\data\dogcat-simplenn\imdb.mat');
disp('Preparing testing data');
folderCat = './DogCatHorse/Testing/Cat/';
folderDog = './DogCatHorse/Testing/Dog/';
%floderHorse = './DogCatHorse/Testing/Horse/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));
%filesHorse = dir(fullfile(folderHorse, '*.jpg'));
featsTest = zeros(length(filesCat) + length(filesDog) , 256);
groundtruthLabel = zeros(length(filesCat) + length(filesDog) , 1);
predictedLabel = zeros(length(filesCat) + length(filesDog) , 1);
for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
im = imread([folderCat filename]);
im_ = rgb2gray(im);
im_ = single(imresize(im_, [64 64])) ;
im_ = im_- imdb.images.data_mean;
im_ = reshape(im_,[64 64 1]);
% run the CNN
res = vl_simplenn(net, im_) ;
% visualize the classification result
scores = squeeze(gather(res(end).x)) ; 
[bestScore, best] = max(scores) ;
figure ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.name{best}, best, bestScore));
predictedLabel(i) = best;
end 
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
im = imread([folderDog filename]);
im_ = rgb2gray(im);
im_ = single(imresize(im_, [64 64])) ;
im_ = im_- imdb.images.data_mean;
im_ = reshape(im_,[64 64 1]);
% run the CNN
res = vl_simplenn(net, im_) ;
% visualize the classification result
scores = squeeze(gather(res(end).x)) ; [bestScore, best] = max(scores) ;
figure ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
    net.meta.classes.name{best}, best, bestScore)) ;
predictedLabel(i+length(filesCat)) = best;
end


disp('Perform Testing');
accurateClassification = 0;
for i = 1:size(featsTest,1)
if predictedLabel(i)>= 0.5
    predictedLabel(i)= 1;
else
    predictedLabel(i) = 0;
end
if(predictedLabel(i) == groundtruthLabel(i))
accurateClassification = accurateClassification + 1;
end
end
accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);