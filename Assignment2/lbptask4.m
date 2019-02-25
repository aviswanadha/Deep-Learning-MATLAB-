close all;
clear all;
clc;

% Testing data
disp('Preparing testing data');
Catfile='./Testing/Cat/';
Dogfile='./Testing/Dog/';
filesCat = dir(fullfile(Catfile, '*.jpg'));
filesDog = dir(fullfile(Dogfile, '*.jpg'));
lbptestfeat = zeros(length(filesCat) + length(filesDog), 256);
groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([Catfile filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    lbptestfeat(i,:) = feat;
    groundtruthLabel(i) = 1;
end

for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([Dogfile filename]);
    img = imresize(img,[256,256]);
    feat = lbp(img);
    lbptestfeat(i + length(filesCat),:) = feat;
    groundtruthLabel(i + length(filesCat)) = 2;
end

% Training Data
disp('Training data for KNN LBP with flipped images');
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

save('lbptestfeatures.mat','lbptestfeat');
save('lbptrainfeatures.mat','lbptrainfeat','lbptrainLabel');
