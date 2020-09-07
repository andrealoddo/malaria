%CNN-Based Image Analysis for Malaria Detection
%comparing performance

% Acquisizione del Data Set
malaria_ds = imageDatastore('C:\Users\Corrado\Documents\Informatica\. Tesi\preprocessed_cell_images','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs, testImgs, validImgs] = splitEachLabel(malaria_ds,0.80,0.10,0.10,'randomized');

%% Alexnet

%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying AlexNet
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer;


% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,layers,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\alexnet_NIH');


%% GoogLeNet

%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying googLeNet
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(googlenet);
lgraph = replaceLayer(lgraph,'loss3-classifier',a);
lgraph = replaceLayer(lgraph,'output',b);


% Opzioni del Training
options = trainingOptions('adam',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'MaxEpochs',10,...
    'ValidationPatience',4,...  
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\googlenet_NIH');


%% InceptionV3

%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([299 299], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([299 299], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([299 299], testImgs,'ColorPreprocessing','gray2rgb');

%Create a network by modifying inceptionV3
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(inceptionv3);
lgraph = replaceLayer(lgraph,'predictions',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);


% Opzioni del Training
options = trainingOptions('adam',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\inceptionv3_NIH');


%% MobileNetV2

%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');

%Create a network by modifying Resnet101
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(mobilenetv2);
lgraph = replaceLayer(lgraph,'Logits',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_Logits',b);


% Opzioni del Training
options = trainingOptions('adam',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,... 
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\mobilenetv2_NIH');


%% ResNet18

%Augmentation e Preprocessing
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);


trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying Resnet18
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet18);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);

% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);


save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\resnet18_NIH');


%% ResNet50


%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');

%Modifica di Resnet50
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet50);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',b);


% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\resnet50_NIH');


%% ResNet101




%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying Resnet101
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet101);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);


% Opzioni del Training
options = trainingOptions('adam',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',400,...
    'MaxEpochs',10,...
    'ValidationPatience',4,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\resnet101_NIH');


%% ShuffleNet



%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying ShuffleNet
a = fullyConnectedLayer(2,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(shufflenet);
lgraph = replaceLayer(lgraph,'node_202',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_node_203',b);


% Opzioni del Training
options = trainingOptions('adam',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'MaxEpochs',10,...     
    'ValidationPatience',4,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);


save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\shufflenet_NIH');


%% SqueezeNet



%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying SqueezeNet
a =  convolution2dLayer([1, 1],2,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
b = classificationLayer('Name','fine');
lgraph = layerGraph(squeezenet);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);
lgraph = replaceLayer(lgraph,'conv10',a);


% Opzioni del Training
options = trainingOptions('adam',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);

save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\squeezenet_NIH');


%% VGG16


%Augmentation
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb','DataAugmentation',imageAugmenter);
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying VGG16
net = vgg16;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer;


% Opzioni del Training
options = trainingOptions('sgdm',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',16,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',600,...
    'MaxEpochs',10,...
    'ValidationPatience',4,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,layers,options);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);


save('C:\Users\User\Documents\Progetto\malaria-master\Classificazione Binaria\risultati\vgg16_NIH');

