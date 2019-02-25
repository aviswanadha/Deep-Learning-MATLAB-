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
img1 = imresize(img,[256,256]);
imgflip = img(:,end:-1:1,:);
figure,imshow(imgflip,[]);
end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img2 = imresize(img,[32,32]);
tiny_image_feat = img2(:);
figure,imshow(img2, []);
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
img3 = imresize(img,[32,32]);
tiny_image_feat = img3(:);
figure,imshow(img3, []);

end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img4 = imresize(img,[32,32]);
tiny_image_feat = img4(:);
figure,imshow(img4, []);

end

