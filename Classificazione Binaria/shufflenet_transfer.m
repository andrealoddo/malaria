%CNN-Based Image Analysis for Malaria Detection

% Acquisizione del Data Set
malaria_ds = imageDatastore('C:\Users\Corrado\Documents\Informatica\. Tesi\cell_images','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs, testImgs, validImgs] = splitEachLabel(malaria_ds,0.80,0.10,0.10,'randomized');

%Augmentation
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
    'MiniBatchSize',8,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',1000,...
    'MaxEpochs',10,...     
    'ValidationPatience',5,...
    'Plots','training-progress',...    
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
confusionchart(truetest,testpreds);