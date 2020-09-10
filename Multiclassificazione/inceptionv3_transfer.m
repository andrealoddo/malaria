%CNN-Based Image Analysis for Malaria Detection

clear  %Clear workspace
clc    %Clear command window

%Clear augmented images
delete_aug_images('s');
delete_aug_images('t');
delete_aug_images('g');

%Dataset acquisition
[trainImgs, testImgs] = create_sets(300);

%Online data augmentation 
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);


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
fracCorrect(h) = numCorrect / numel(testpreds)
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;