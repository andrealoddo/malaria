function [trainImgs, testImgs,ds_r,aug_ds_t_complete,aug_ds_s_complete,aug_ds_g_complete] = create_sets_special(n)
%create_sets crea un training set con n immagini per classe. Le immagini di
%classe r sono un sottoinsieme della classe originale. Le immagini di
%classe s e t sono sottoposte a un meccanismo di oversampling tramite
%augmentation fino a essere n per classe nel training set. Nel test set
%sono contenute immagini (il 20% di ogni classe) non sottoposte ad
%augmentation e che la rete non prende in input durante il training.


%CREAZIONE DEL TEST SET

%Calcolo della quantità di immagini di classe r necessaria affinché nel
%training set ci siano n immagini (e il restante 20% nel test set)
m = n*10/8;

%Sottoinsieme delle immagini di classe r
path = 'C:\Users\User\Documents\Progetto\Dataset\Falciparum';
ds_r = imageDatastore([path '\r'],'IncludeSubfolders',true,'LabelSource','foldernames');
ds_r = shuffle(ds_r);
ds_r_sub = subset(ds_r,1:m);


%Il 20% delle immagini di ogni classe viene inserito nel test set. Le
%restanti nel training set
x=m*1/5; 
ds_r_sub = shuffle(ds_r_sub);
ds_r_test = subset(ds_r_sub,1:x);
x=x+1;
ds_r_training = subset(ds_r_sub,x:m);

ds_s = imageDatastore([path '\s'],'IncludeSubfolders',true,'LabelSource','foldernames');
ds_s = shuffle(ds_s);
ds_s_test = subset(ds_s,1:4);
ds_s_training = subset(ds_s,5:18);

ds_t = imageDatastore([path '\t'],'IncludeSubfolders',true,'LabelSource','foldernames');
ds_t = shuffle(ds_t);
ds_t_test = subset(ds_t,1:8);
ds_t_training = subset(ds_t,9:42);

ds_g = imageDatastore([path '\g'],'IncludeSubfolders',true,'LabelSource','foldernames');
ds_g = shuffle(ds_g);
ds_g_test = subset(ds_g,1:2);
ds_g_training = subset(ds_g,3:7);

%Creazione del test set contente il 20% di ogni classe
testImgs = merge_sets(ds_r_test,ds_s_test,ds_t_test,ds_g_test);


%Per costruire il training set è necessario aumentare la quantità delle
%immagini di classe s e t, n per classe.
[aug_ds_s,aug_ds_s_complete] = oversampling_special(n,13,ds_s_training,'s');
[aug_ds_t,aug_ds_t_complete] = oversampling_special(n,33,ds_t_training,'t');
[aug_ds_g,aug_ds_g_complete] = oversampling_special(n,4,ds_g_training,'g');

%Formazione delle n immagini della classe da aggiungere nel training set
ds_s_training = merge_sets(ds_s_training,aug_ds_s); 
ds_t_training = merge_sets(ds_t_training,aug_ds_t);
ds_g_training = merge_sets(ds_g_training,aug_ds_g);

%Formazione del training set unendo le n immagini per classe. Il training
%set ha dimensione 3*n
trainImgs = merge_sets(ds_t_training,ds_s_training,ds_r_training,ds_g_training);

end

