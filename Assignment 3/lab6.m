function [net, info] = Lab6c(varargin)
%%% Demonstrates MatConvNet on a custom network
addpath(genpath('./matconvnet/'));
run(fullfile(fileparts(mfilename('fullpath')),...
  'matconvnet', 'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['dogcat-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'dogcat') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

% if isempty(opts.network)
%   net = cnn_dogcat_init('batchNormalization', opts.batchNormalization, ...
%     'networkType', opts.networkType) ;
% else
%   net = opts.network ;
%   opts.network = [] ;
% end


rng('default');
rng(0) ;

f=1/100 ;

net.layers = {} ;

net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(9,9,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;
                       
%% reduce this window
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [7 7], ...
                           'stride', 3, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'relu') ;
%% insert a deeper layer
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,1,10, 'single'), zeros(1, 10, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'conv1') ;
                       
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ... %% can;t chage here
                           'pad', 0) ;
                       
%% Insert dropout layer
net.layers{end+1} = struct('type','dropout','rate',0.3); %%0.5 for no insertion
%%rate can be changed                            
                       
%% adjust final layer
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(6,6,10,3, 'single'), zeros(1, 3, 'single')}}, ...
                           'stride', 1, ...
                           'pad', 0, ...
                           'name', 'fc1') ;
                     
% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [64 64 1] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 40 ;
net.meta.trainOpts.batchSize = 100 ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

folderCat = './DogCatHorse/Training/Cat/';
folderDog = './DogCatHorse/Training/Dog/';
folderHorse = './DogCatHorse/Training/Horse/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));
filesHorse = dir(fullfile(folderHorse, '*.jpg'));

feats = zeros(64,64,length(filesCat) + length(filesDog)+length(filesHorse));
labels = zeros(length(filesCat) + length(filesDog) + length(filesHorse),1);

for i = 1:length(filesCat) 
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = rgb2gray(img);
    img = imresize(img,[64,64]);
    feats(:,:,i) = img;
    labels(i) = 0;
end

for i = 1:length(filesDog) 
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = rgb2gray(img);
    img = imresize(img,[64,64]);
    feats(:,:,i + length(filesCat)) = img;
    labels(i + length(filesCat)) = 1;
end

for i = 1:length(filesHorse) 
    disp(i);
    filename = filesHorse(i,1).name;
    img = imread([folderHorse filename]);
    img = rgb2gray(img);
    img = imresize(img,[64,64]);
    feats(:,:,i + length(filesCat)+ length(filesDog)) = img;
    labels(i + length(filesCat)+length(filesDog)) = 2;
end

%% Preparing testing data
disp('Preparing testing data');
folderCat = './DogCatHorse/Testing/Cat/';
folderDog = './DogCatHorse/Testing/Dog/';
folderHorse = './DogCatHorse/Testing/Horse/';

filesCat = dir(fullfile(folderCat, '*.jpg'));
filesDog = dir(fullfile(folderDog, '*.jpg'));
filesHorse = dir(fullfile(folderHorse, '*.jpg'));

featsTest = zeros(64,64, length(filesCat) + length(filesDog)+length(filesHorse));
global groundtruthLabel;
groundtruthLabel= zeros(length(filesCat) + length(filesDog)+length(filesHorse), 1);
% predictedLabel = zeros(length(filesCat) + length(filesDog), 1);

for i = 1:length(filesCat) 
    disp(i);
    filename = filesCat(i,1).name;
    img = imread([folderCat filename]);
    img = rgb2gray(img);
    img = imresize(img,[64,64]);
%     feat = imglbp(img);
    featsTest(:,:,i) = img;
    groundtruthLabel(i) = 0;
end

for i = 1:length(filesDog) 
    disp(i);
    filename = filesDog(i,1).name;
    img = imread([folderDog filename]);
    img = rgb2gray(img);
    img = imresize(img,[64,64]);
    featsTest(:,:,i + length(filesCat)) = img;
    groundtruthLabel(i + length(filesCat)) = 1;
end

for i = 1:length(filesHorse) 
    disp(i);
    filename = filesHorse(i,1).name;
    img = imread([folderHorse filename]);
    img = rgb2gray(img);
    img = imresize(img,[64,64]);
    featsTest(:,:,i + length(filesCat)+length(filesDog)) = img;
    groundtruthLabel(i + length(filesCat)+length(filesDog)) = 2;
end

set = [ones(1,numel(labels)) 3*ones(1,numel(groundtruthLabel))];
data = single(reshape(cat(3, feats, featsTest),64,64,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, labels', groundtruthLabel') ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:2,'uniformoutput',false) ;

mkdir(opts.expDir) ;
save(opts.imdbPath, '-struct', 'imdb') ;
  
net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:3,'UniformOutput',false) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),64,64,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),64,64,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),64,64,1,[]));
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;
