%CNN-Based Image Analysis for Malaria Detection
clear
clc


%Online data augmentation 
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);


%Training set e Test set
path = strcat('C:\Users\User\Documents\GitHub\malaria_networks\MP-IDB. Falciparum\alexnet_MP-IDB_test100');
load(path,'trainImgs')
load(path,'testImgs')
load(path,'validImgs')

   
%% Alexnet 

%Alexnet fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\alexnet_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
net = malarianet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(4);
layers(end) = classificationLayer;

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'MaxEpochs',10,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,layers,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\alexnet')

%% ResNet18 

%ResNet18 fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\resnet18_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5


trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\resnet18')

%% ResNet50 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\resnet50_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5


trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\resnet50')

%% GoogLeNet 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\googlenet_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5



trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\googlenet')

%% Resnet101 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\resnet101_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5



trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying Resnet101
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);


% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'MaxEpochs',10,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',40,...
    'Verbose',false,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu');


%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\resnet101')

%% ShuffleNet 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\shufflenet_NIH_con_preprocessing');
load(pathNet,'malarianet')


for h=1:5

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\shufflenet')

%% SqueezeNet 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\squeezenet_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);
testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');

%Modifying pretrained network
a =  convolution2dLayer([1, 1],4,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'fine',b);
lgraph = replaceLayer(lgraph,'new_conv',a);


%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\squeezenet')

%% MobileNetV2 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\mobilenetv2_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');

%Create a network by modifying Resnet101
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);

% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'MaxEpochs',10,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'Verbose',false,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\mobilenetv2')

%% inceptionv3 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\inceptionv3_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5

trainImgs_a = augmentedImageDatastore([299 299], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([299 299], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([299 299], testImgs,'ColorPreprocessing','gray2rgb');

%Create a network by modifying inceptionV3
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(malarianet);
lgraph = replaceLayer(lgraph,'FCL',a);
lgraph = replaceLayer(lgraph,'fine',b);


% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'MaxEpochs',10,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',40,...
    'Verbose',false,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu');


%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\inceptionv3')

%% vgg16 

%net fine-tuned on NIH
pathNet = strcat('C:\Users\User\Documents\GitHub\malaria_networks\NIH\reti addestrate su NIH con preprocessing\vgg16_NIH_con_preprocessing');
load(pathNet,'malarianet')

for h=1:5
trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying VGG16
net = malarianet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(4);
layers(end) = classificationLayer;


% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',16,...
    'MaxEpochs',10,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',80,...
    'Verbose',false,...
    'Shuffle','every-epoch',...
    'ExecutionEnvironment','gpu');


%Training
[malarianet,info] = trainNetwork(trainImgs_a,layers,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save('C:\Users\User\Documents\GitHub\malaria_networks\NIH + MPIDB\vgg16')
