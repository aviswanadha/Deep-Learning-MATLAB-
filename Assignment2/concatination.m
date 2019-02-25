close all;
clear all;
clc;

load 'lbptestfeatures.mat';
load 'lbptrainfeatures.mat';
load 'fctestfeatures.mat';
load 'fctrainfeatures.mat';

%testing data normaization
lbptestfeat=lbptestfeat./norm(lbptestfeat);
fctestfeat=fctestfeat/norm(fctestfeat);

featsTest=[lbptestfeat fctestfeat];

%training data norm
lbptrainfeat=lbptrainfeat/norm(lbptrainfeat);
fctrainfeat=fctrainfeat/norm(fctrainfeat);

featsTrain=[lbptrainfeat fctrainfeat];

% Accuracy with ability to change k value

disp('Accuracy for both LBP and FC');
accurateClassification = 0;
for i = 1:size(featsTest,1)
feat = featsTest(i,:);
dists = pdist2(feat,featsTrain,'minkowski',3);
[val, idx] = min(dists);
predictedLabel(i) = lbptrainLabel(idx);
if(predictedLabel(i) == groundtruthLabel(i))
accurateClassification = accurateClassification + 1;
end
end
accuracy = accurateClassification/length(groundtruthLabel);
disp(['The accuracy:' num2str(accuracy * 100) '%']);
