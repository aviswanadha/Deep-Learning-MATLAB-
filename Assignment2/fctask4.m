close all;
clear all;
clc;

% Testing data
disp('Preparing testing data');
Catfile='./Testing/Cat/';
Dogfile='./Testing/Dog/';
filesCat = dir(fullfile(Catfile, '*.jpg'));
filesDog = dir(fullfile(Dogfile, '*.jpg'));
fctestfeat = zeros(length(filesCat) + length(filesDog), 4096);
groundtruthLabel = zeros(length(filesCat) + length(filesDog), 1);
predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([Catfile filename]);
    img = imresize(img,[227,227]);
    feat = fcfun(img);
    fctestfeat(i,:) = feat;
    groundtruthLabel(i) = 1;
end

for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([Dogfile filename]);
    img = imresize(img,[227,227]);
    feat = fcfun(img);
    fctestfeat(i + length(filesCat),:) = feat;
    groundtruthLabel(i + length(filesCat)) = 2;
end

disp('Training data for KNN FC with flipped images ');
Catfile='./Training3/Cat/';
Dogfile='./Training3/Dog/';
filesCat = [dir(fullfile(Catfile, '*.jpg')); dir(fullfile(Catfile, '*.jfif')); dir(fullfile(Catfile, '*.png'));];
filesDog = [dir(fullfile(Dogfile, '*.jpg')); dir(fullfile(Dogfile, '*.jfif')); dir(fullfile(Dogfile, '*.png'));];
fctrainfeat=zeros(length(filesCat)+length(filesDog),4096);
fctrainLabel=zeros(length(filesCat)+length(filesDog),1);

for i = 1:length(filesCat)
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([Catfile filename]);
    img = imresize(img,[227,227]);
    feat = fcfun(img);
    fctrainfeat(i,:) = feat;
    fctrainLabel(i) = 1;
end

for i = 1:length(filesDog)
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([Dogfile filename]);
    img = imresize(img,[227,227]);
    feat = fcfun(img);
    fctrainfeat(i + length(filesCat),:) = feat;
    fctrainLabel(i + length(filesCat)) = 2;
end


save('fctestfeatures.mat','fctestfeat','groundtruthLabel');
save('fctrainfeatures.mat','fctrainfeat','fctrainLabel');