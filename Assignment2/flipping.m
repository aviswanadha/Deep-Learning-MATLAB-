close all;
clear all;
clc;
disp('Preparing training data');
folderCat = './Training2/Cat/';
folderDog = './Training2/Dog/';
filesCat = [dir(fullfile(folderCat, '*.jpg')); dir(fullfile(folderCat, '*.jfif')); dir(fullfile(folderCat, '*.png'));];
filesDog = [dir(fullfile(folderDog, '*.jpg')); dir(fullfile(folderDog, '*.jfif')); dir(fullfile(folderDog, '*.png'));];

for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
img = imread([folderCat filename]);
img = imresize(img,[256,256]);
imgflip = img(:,end:-1:1,:);
figure,imshow(imgflip); 
%imsave;

end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img = imresize(img,[256,256]);
imgflip = img(:,end:-1:1,:);
figure,imshow(imgflip); 
%imsave;
end

disp('Preparing testing data');
folderCat = './Testing/Cat/';
folderDog = './Testing/Dog/';
filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));

for i = 1:length(filesCat)
disp(i);
filename = filesCat(i,1).name;
img = imread([folderCat filename]);
img1 = imresize(img,[256,256]);
tiny_image_feat = img1(:);
figure,imshow(img1, []);
end
for i = 1:length(filesDog)
disp(i);
filename = filesDog(i,1).name;
img = imread([folderDog filename]);
img2 = imresize(img,[256,256]);
tiny_image_feat = img2(:);
figure,imshow(img2, []);
end
