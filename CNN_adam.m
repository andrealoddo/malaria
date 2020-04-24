%CNN-Based Image Analysis for Malaria Detection

% Acquisizione del Data Set
malaria_ds = imageDatastore('C:\Users\Corrado\Documents\Informatica\. Tesi\cell_images','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs, testImgs, validImgs] = splitEachLabel(malaria_ds,0.80,0.10,0.10,'randomized');

%Augmentation
trainImgs_a = augmentedImageDatastore([44 44], trainImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([44 44], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([44 44], testImgs,'ColorPreprocessing','gray2rgb');


% Creazione della rete
layers = [
    imageInputLayer([44 44 3]);
    convolution2dLayer(5, 32);
    reluLayer;    
    convolution2dLayer(5, 32);
    reluLayer;   
    maxPooling2dLayer(5);
    convolution2dLayer(5, 64);
    reluLayer;    
    convolution2dLayer(3, 64);
    averagePooling2dLayer(3);
    convolution2dLayer(5, 128);
    reluLayer;    
    convolution2dLayer(4, 256);
    fullyConnectedLayer(256);
    fullyConnectedLayer(256);
    fullyConnectedLayer(2);
    softmaxLayer();
    classificationLayer();
];

% Opzioni del Training
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',128,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',50,...
    'ValidationPatience',4,...
    'MaxEpochs',15,...
     'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',6, ...
    'Plots','training-progress',...    
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,layers,options);
trainAcc = info.TrainingAccuracy(end);
trainLoss = info.TrainingLoss(end);
validAcc = info.ValidationAccuracy(end);
validLoss = info.ValidationLoss(end);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds)
