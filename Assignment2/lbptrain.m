close all;
clear all;
clc;

% Training Data
disp('Training flip data for KNN ');
Catfile='./Training3/Cat/';
Dogfile='./Training3/Dog/';
filesCat = [dir(fullfile(Catfile, '*.jpg')); dir(fullfile(Catfile, '*.jfif')); dir(fullfile(Catfile, '*.png'));];
filesDog = [dir(fullfile(Dogfile, '*.jpg')); dir(fullfile(Dogfile, '*.jfif')); dir(fullfile(Dogfile, '*.png'));];
lbptrainfeat=zeros(length(filesCat)+length(filesDog),256);
lbptrainLabel=zeros(length(filesCat)+length(filesDog),1);

for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([Catfile filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    lbptrainfeat(i,:) = feat;
    lbptrainLabel(i) = 1;
end


for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([Dogfile filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    lbptrainfeat(i + length(filesCat),:) = feat;
    lbptrainLabel(i + length(filesCat)) = 2;
end

save('lbptrainfeatures.mat','lbptrainfeat','lbptrainLabel');