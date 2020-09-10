function [aug_ds_sub,aug_ds] = oversampling_special(n,class_n,ds_training,char)
%oversampling crea delle augmented images necessarie per aumentare la
%quantità di immagini per classe nel training set


%Numero di copie che ogni immagine di una classe deve avere per poter
%aumentare la quantità a n
num_aug = round(300/class_n)+1;

%Creazione delle copie delle immagini: per ognuna delle class_n immagini,
%crea num_aug copie
for idx = 1:class_n
    immagine = readimage(ds_training,idx);
    path = ds_training.Files{idx};
    aug(immagine,path,char,num_aug);
end

%Si estrae il numero necessario di augmented images per far sì che il
%numero di immagini per classe nel training set sia n
path = 'C:\Users\User\Documents\Progetto\Dataset\F_A\';
aug_ds = imageDatastore([path char],'IncludeSubfolders',true,'LabelSource','foldernames');
aug_ds = shuffle(aug_ds);
ns=n-class_n-1;
aug_ds_sub = subset(aug_ds,1:ns);

end

