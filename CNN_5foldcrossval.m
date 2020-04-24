%CNN-Based Image Analysis for Malaria Detection

% Acquisizione del Data Set
malaria_ds = imageDatastore('C:\Users\Corrado\Documents\Informatica\. Tesi\cell_images','IncludeSubfolders',true,'LabelSource','foldernames');

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
options = trainingOptions('sgdm','InitialLearnRate',0.00001,'Verbose',false,'MiniBatchSize',32,'MaxEpochs',10,'Shuffle','every-epoch','Momentum',0.9,'L2Regularization',0.000001,'ExecutionEnvironment','gpu','Plots','training-progress');


% 5-fold cross validation
[imd1, imd2, imd3, imd4, imd5] = splitEachLabel(malaria_ds, 0.2,0.2,0.2,0.2,0.2,'randomized');
 
partStores{1} = imd1.Files;
partStores{2} = imd2.Files;
partStores{3} = imd3.Files;
partStores{4} = imd4.Files;
partStores{5} = imd5.Files;

k = 5;
idx = crossvalind('Kfold', k, k);

fracCorrect = zeros(1,5);
trainAcc = zeros(1,5);
trainLoss = zeros(1,5);

for i = 1:5
    
test_idx = (idx == i);
train_idx = ~test_idx;

trainImgs = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true,'FileExtensions','.png', 'LabelSource', 'foldernames');
testImgs = imageDatastore(partStores{test_idx}, 'IncludeSubfolders', true,'FileExtensions','.png', 'LabelSource', 'foldernames');

%Augmentation
trainImgs_a = augmentedImageDatastore([100 100], trainImgs,'ColorPreprocessing','gray2rgb');
testImgs_a = augmentedImageDatastore([100 100], testImgs,'ColorPreprocessing','gray2rgb');

% Training
[malarianet,info] = trainNetwork(trainImgs_a,layers,options);
trainAcc(i) = info.TrainingAccuracy(end);
trainLoss(i) = info.TrainingLoss(end);

% Classificazione delle immagini
testpreds = classify(malarianet,testImgs_a);

% Validazione della rete
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(i) = numCorrect / numel(testpreds);
confusionchart(truetest,testpreds)

end

fprintf("\n\n\t\t\t\t\tRISULTATI\n")
fprintf("_____________________________________________________\n")
fprintf("k\ttest accuracy\ttraining accuracy\ttraining loss\n")
fprintf("-----------------------------------------------------\n")

for i = 1:5
    fprintf("%d\t %.3f\t\t\t%.3f\t\t\t\t%.3f\n",i,fracCorrect(i),trainAcc(i),trainLoss(i))
end