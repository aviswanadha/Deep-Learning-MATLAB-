function [ feat ] = fcfun(im)
%DLFUNCTION Summary of this function goes here
%   Detailed explanation goes here

%Add MatConvNet to the project
run './matconvnet/matlab/vl_setupnn'

%load the 227MB pre-trained CNN
net = load('imagenet-caffe-alex.mat');

im_ = single(im) ; 
im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
for j = 1:3
im_(:,:,j) = im_(:,:,j)- net.meta.normalization.averageImage(j);
end

% run the CNN
res = vl_simplenn(net, im_) ;

feat=res(20).x;
feat=feat(:);

end



