%CNN-Based Image Analysis for Malaria Detection

clear  %Clear workspace
clc    %Clear command window

%Clear augmented images
delete_aug_images('s');
delete_aug_images('t');
delete_aug_images('g');

%Dataset acquisition
[trainImgs, testImg

%Online data augmentation 
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);s] = create_sets(300);

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