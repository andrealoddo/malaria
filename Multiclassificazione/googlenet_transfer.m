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
fracCorrect(h) = numCorrect / numel(testpreds)
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;