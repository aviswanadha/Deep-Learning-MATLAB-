close all;
clear all;
clc;
disp('Preparing training data');
folderCat = './Training3/Cat/';
folderDog = './Training3/Dog/';
filesCat = [dir(fullfile(folderCat, '*.jpg')); dir(fullfile(folderCat, '*.jfif')); dir(fullfile(folderCat, '*.png'));];
filesDog = [dir(fullfile(folderDog, '*.jpg')); dir(fullfile(folderDog, '*.jfif')); dir(fullfile(folderDog, '*.png'));];
feats = zeros(length(filesCat) + length(filesDog), 256);
labels = zeros(length(filesCat) + length(filesDog),1);
for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
img = imread([folderCat filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
feats(i,:) = feat;
labels(i) = 1;
end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
feats(i + length(filesCat),:) = feat;
labels(i + length(filesCat)) = 2;
end
disp('Preparing testing data');
folderCat = './Testing/Cat/';
folderDog = './Testing/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));
featsTest = zeros(length(filesCat) + length(filesDog), 256);
groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);
for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
img = imread([folderCat filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
featsTest(i,:) = feat;
groundtruthLabel(i) = 1;
end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
featsTest(i + length(filesCat),:) = feat;
groundtruthLabel(i + length(filesCat)) = 2;
end
disp('Performing testing');
accurateClassification = 0;
for i = 1:size(featsTest,1)
feat = featsTest(i,:);
dists = pdist2(feat,feats,'minkowski',3);
[val, idx] = min(dists);
predictedLabel(i) = labels(idx);
if(predictedLabel(i) == groundtruthLabel(i))
accurateClassification = accurateClassification + 1;
end
end
accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);