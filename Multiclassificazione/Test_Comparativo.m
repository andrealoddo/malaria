%CNN-Based Image Analysis for Malaria Detection

delete_aug_images('s');
delete_aug_images('t');

tests = [100 200 300];

%Online data augmentation 
imageAugmenter = imageDataAugmenter('RandRotation',[-90,90],...
    'RandXTranslation',[-10 10],'RandYTranslation',[-10 10],...
    'RandXReflection',true,'RandYReflection',true);

for l=1:3
   x = tests(l); 
   
if l == 1
    [trainImgs,testImgs,poolR,poolT,poolS] = create_sets_special(x);
    shuffle(trainImgs);

    validImgs = subset(trainImgs,1:10);
    trainImgs = subset(trainImgs,11:300);
    shuffle(trainImgs);

end

if l == 2
    ds_r_sub = subset(poolR,176:275);
    ds_t_sub = subset(poolT,101:200);
    ds_s_sub = subset(poolS,101:200);
    
    add_images = merge_sets(ds_r_sub,ds_t_sub,ds_s_sub); 
    val_sub = subset(add_images,1:10);
    validImgs = merge_sets(validImgs,val_sub);
    
    train_sub = subset(add_images,11:300);
    trainImgs = merge_sets(trainImgs,train_sub); 
    
    
end

 if l == 3
    ds_r_sub = subset(poolR,276:375);
    ds_t_sub = subset(poolT,201:300);
    ds_s_sub = subset(poolS,201:300);

    add_images = merge_sets(ds_r_sub,ds_t_sub,ds_s_sub); 
    val_sub = subset(add_images,1:10);
    validImgs = merge_sets(validImgs,val_sub);
    
    train_sub = subset(add_images,11:300);
    trainImgs = merge_sets(trainImgs,train_sub); 
    
 end
   
for h=1:5
      


trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);
testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');

%Modifying pretrained network
a =  convolution2dLayer([1, 1],3,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,"Name",'new_conv');
b = classificationLayer('Name','fine');
lgraph = layerGraph(squeezenet);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);
lgraph = replaceLayer(lgraph,'conv10',a);


%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'Verbose',false,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',10,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

%Classification
testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = nnz(testpreds == truetest);
fracCorrect(h) = numCorrect / numel(testpreds)

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save(['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 4\squeezenet' x])
%CNN-Based Image Analysis for Malaria Detection

for h=1:5

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(3,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet18);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
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

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save(['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 4\resnet18' x])
%CNN-Based Image Analysis for Malaria Detection

for h=1:5

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(3,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(resnet50);
lgraph = replaceLayer(lgraph,'fc1000',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',8,...
    'MaxEpochs',10,...
    'Shuffle','every-epoch',...
    'ValidationFrequency',30,...
    'Verbose',false,...
    'ExecutionEnvironment','gpu');

%Training
[malarianet,info] = trainNetwork(trainImgs_a,lgraph,options);

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
save(['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 4\resnet50' x])
%CNN-Based Image Analysis for Malaria Detection

for h=1:5
    
g = gpuDevice(1);
M = gpuArray(magic(4));
reset(g);
clear M
trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(3,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(shufflenet);
lgraph = replaceLayer(lgraph,'node_202',a);
lgraph = replaceLayer(lgraph,'ClassificationLayer_node_203',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
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

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save(['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 4\shufflenet' x])

%CNN-Based Image Analysis for Malaria Detection

for h=1:5
    

trainImgs_a = augmentedImageDatastore([224 224], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([224 224], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
a = fullyConnectedLayer(3,'Name','FCL');
b = classificationLayer('Name','fine');
lgraph = layerGraph(googlenet);
lgraph = replaceLayer(lgraph,'loss3-classifier',a);
lgraph = replaceLayer(lgraph,'output',b);

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',32,...
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
save(['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 4\googlenet' x])

%CNN-Based Image Analysis for Malaria Detection

for h=1:5

trainImgs_a = augmentedImageDatastore([227 227], trainImgs,'ColorPreprocessing','gray2rgb',...
    'DataAugmentation',imageAugmenter);

testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
validImgs_a = augmentedImageDatastore([227 227], validImgs,'ColorPreprocessing','gray2rgb');


%Modifying pretrained network
net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(3);
layers(end) = classificationLayer;

%Training options
options = trainingOptions('adam',...
    'InitialLearnRate',1e-4,...
    'MiniBatchSize',8,...
    'MaxEpochs',10,...
    'ValidationData',validImgs_a,...
    'ValidationFrequency',30,...
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
fracCorrect(h) = numCorrect / numel(testpreds)

end

media=0;
for h=1:5
   media=media+fracCorrect(h); 
end

media=media/5;
save(['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 4\alexnet' x])

   
   
   
end
