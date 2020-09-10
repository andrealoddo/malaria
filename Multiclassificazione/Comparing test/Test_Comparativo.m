%CNN-Based Image Analysis for Malaria Detection
clear
clc

%Clear augmented images
delete_aug_images('s');
delete_aug_images('t');
delete_aug_images('g');

%numero di immagini per classe nel training set
tests = [100 200 300];

%Online data augmentation 
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

%Tre cicli di test: in ogni ciclo si hanno rispettivamente 100, 200 o 300
%immagini per classe nel training set
for l=1:3
   x = tests(l); 
   
if l == 1   %100 immagini per classe nel training set
    
    %Creazione di Training set e Test set
    [trainImgs,testImgs,poolR,poolT,poolS,poolG] = create_sets_special(x);
    shuffle(trainImgs);
    
    %Creazione del Validation set: il 10% del training set
    validImgs = subset(trainImgs,1:40);
    trainImgs = subset(trainImgs,41:400);
    shuffle(trainImgs);

end

if l == 2  %200 immagini per classe
    
    %Si estraggono 100 immagini non utilizzate per ogni classe
    ds_r_sub = subset(poolR,126:225);
    ds_t_sub = subset(poolT,101:200);
    ds_s_sub = subset(poolS,101:200);
    ds_g_sub = subset(poolG,101:200);
    
    add_images = merge_sets(ds_r_sub,ds_t_sub,ds_s_sub,ds_g_sub); 
    
    %Si aggiungono immagini nel validation set
    val_sub = subset(add_images,1:40);
    validImgs = merge_sets(validImgs,val_sub);
    
    %Si aggiungono immagini nel training set
    train_sub = subset(add_images,41:400);
    trainImgs = merge_sets(trainImgs,train_sub); 
    
    
end

 if l == 3  %300 immagini per classe
     
    %Si estraggono 100 immagini non utilizzate per ogni classe
    ds_r_sub = subset(poolR,226:325);
    ds_t_sub = subset(poolT,201:300);
    ds_s_sub = subset(poolS,201:300);
    ds_g_sub = subset(poolG,201:300);

    add_images = merge_sets(ds_r_sub,ds_t_sub,ds_s_sub,ds_g_sub); 
    
    %Si aggiungono immagini nel validation set    
    val_sub = subset(add_images,1:40);
    validImgs = merge_sets(validImgs,val_sub);
    
    %Si aggiungono immagini nel training set
    train_sub = subset(add_images,41:400);
    trainImgs = merge_sets(trainImgs,train_sub); 
    
 end
   
%% Alexnet 

for h=1:5

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
net = alexnet;
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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\alexnet' '_MP-IDB_test' int2str(x)])

%% ResNet18 

for h=1:5


trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet18);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);

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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\resnet18' '_MP-IDB_test' int2str(x)])

%% ResNet50 

for h=1:5


trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet50);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',b);

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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\resnet50' '_MP-IDB_test' int2str(x)])

%% GoogLeNet 

for h=1:5



trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(googlenet);
lgraph = replaceLayer(lgraph,'loss3-classifier',a);
lgraph = replaceLayer(lgraph,'output',b);

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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\googlenet' '_MP-IDB_test' int2str(x)])

%% Resnet101 

for h=1:5



trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying Resnet101
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet101);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);


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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\resnet101' '_MP-IDB_test' int2str(x)])

%% ShuffleNet 


for h=1:5

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(shufflenet);
lgraph = replaceLayer(lgraph,'node_202',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_node_203',b);

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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\shufflenet' '_MP-IDB_test' int2str(x)])

%% SqueezeNet 

for h=1:5

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);
testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');

%Modifying pretrained network
a =  convolution2dLayer([1, 1],4,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
b = classificationLayer('Name','fine');
lgraph = layerGraph(squeezenet);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);
lgraph = replaceLayer(lgraph,'conv10',a);


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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\squeezenet' '_MP-IDB_test' int2str(x)])

%% MobileNetV2 

for h=1:5

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');

%Create a network by modifying Resnet101
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(mobilenetv2);
lgraph = replaceLayer(lgraph,'Logits',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',b);

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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\mobilenetv2' '_MP-IDB_test' int2str(x)])

%% inceptionv3 

for h=1:5

trainImgs_a = augmentedImageDatastore([299 299], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([299 299], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([299 299], testImgs,'ColorPreprocessing','gray2rgb');

%Create a network by modifying inceptionV3
a = fullyConnectedLayer(4,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(inceptionv3);
lgraph = replaceLayer(lgraph,'predictions',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);


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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\inceptionv3' '_MP-IDB_test' int2str(x)])

%% vgg16 

for h=1:5
trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying VGG16
net = vgg16;
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
save(['C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\vgg16' '_MP-IDB_test' int2str(x)])

end