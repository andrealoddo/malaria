%CNN-Based Image Analysis for Malaria Detection

% Acquisizione del Data Set
malaria_ds = imageDatastore('C:\Users\Corrado\Documents\Informatica\. Tesi\cell_images','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs, testImgs, validImgs] = splitEachLabel(malaria_ds,0.80,0.10,0.10,'randomized');

%Augmentation
trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');


%Create a network by modifying AlexNet
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(2);
layers(end) = classificationLayer;


% Opzioni del Training
options = trainingOptions('sgdm',...
     'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',200,...
    'ValidationPatience',4,...
    'MaxEpochs',10,...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',2, ...
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

