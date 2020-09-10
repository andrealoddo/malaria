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
fracCorrect(h) = numCorrect / numel(testpreds)
confusionchart(truetest,testpreds);

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;