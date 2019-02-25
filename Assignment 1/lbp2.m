close all;
clear all;
clc;
disp('Preparing training data');
folderCat = './Training1/Cat/';
folderDog = './Training1/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
img = imread([folderCat filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
figure,bar(feat);   

end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
figure,bar(feat);

end
disp('Preparing testing data');
folderCat = './Testing1/Cat/';
folderDog = './Testing1/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
img = imread([folderCat filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
figure,bar(feat);
end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img = imresize(img,[256,256]);
feat = lbp(img);
figure,bar(feat);
end


