%CNN-Based Image Analysis for Malaria Detection

% Acquisizione del Data Set
malaria_ds = imageDatastore('C:\Users\Corrado\Documents\Informatica\. Tesi\cell_images','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs, testImgs, validImgs] = splitEachLabel(malaria_ds,0.80,0.10,0.10,'randomized');

%Augmentation
trainImgs_a = augmentedImageDatastore([100 100], trainImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([100 100], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([100 100], testImgs,'ColorPreprocessing','gray2rgb');


% Creazione della rete
layers = [
    imageInputLayer([100 100 3]);
    convolution2dLayer(3, 32);
    reluLayer;    
    maxPooling2dLayer(2);    
    convolution2dLayer(3, 32);
    reluLayer;   
    maxPooling2dLayer(2);    
    convolution2dLayer(3, 64);
    reluLayer;   
    maxPooling2dLayer(2);   
    fullyConnectedLayer(64);
    reluLayer;
    dropoutLayer(0.5);
    fullyConnectedLayer(2);
    softmaxLayer();
    classificationLayer();
];

% Opzioni del Training
options = trainingOptions('sgdm',...
    'InitialLearnRate',1e-6,...
    'MiniBatchSize',64,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',100,...
    'ValidationPatience',4,...
    'MaxEpochs',20,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',6, ...
    'Plots','training-progress',...    
    'Momentum',0.9, ...
    'L2Regularization', 1e-6,...
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
