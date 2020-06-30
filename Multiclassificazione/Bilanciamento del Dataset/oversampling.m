function [aug_ds] = oversampling(n,class_n,ds_training,char)
%oversampling crea delle augmented images necessarie per aumentare la
%quantità di immagini per classe nel training set


%Numero di copie che ogni immagine di una classe deve avere per poter
%aumentare la quantità a n
num_aug = round(n/class_n)+1;

%Creazione delle copie delle immagini
for idx = 1:class_n
    immagine = readimage(ds_training,idx);
    path = ds_training.Files{idx};
    aug(immagine,path,char,num_aug);
end

%Si estrae il numero necessario di augmented images per far sì che il
%numero di immagini per classe nel training set sia n
path = 'C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\F A\';
aug_ds = imageDatastore([path char],'IncludeSubfolders',true,'LabelSource','foldernames');
aug_ds = shuffle(aug_ds);
ns=n-class_n;
aug_ds = subset(aug_ds,1:ns);

end

