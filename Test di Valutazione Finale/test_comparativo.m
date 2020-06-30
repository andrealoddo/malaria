% Questo test utilizza le reti allenate con il dataset NIH e svolge un test
% di classificazione sul dataset MP-IDB


tests = ["alexnet", "googlenet", "resnet18", "resnet50", "shufflenet", "squeezenet"];

for i=1:6
   x = tests(i); 
   
%Loading dataset and net
pathDataStore = 'C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\F (crops)';
pathNet = strcat('C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 3\', x, '_workspace');
pathSave = strcat('C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\Risultati workspaces 5\', x, '_workspace');
load(pathNet,'malarianet')
testImgs = imageDatastore(pathDataStore,'IncludeSubfolders',true,'LabelSource','foldernames');


if i==1 || i==6
    testImgs_a = augmentedImageDatastore([227 227], testImgs,'ColorPreprocessing','gray2rgb');
else
    testImgs_a = augmentedImageDatastore([224 224], testImgs,'ColorPreprocessing','gray2rgb');
end

%Classify
 testpreds = classify(malarianet,testImgs_a);

%Validation
truetest = testImgs.Labels;
numCorrect = num_pred_corr(testpreds,truetest);
fracCorrect = numCorrect / numel(testpreds);



%Save
save(pathSave)
end