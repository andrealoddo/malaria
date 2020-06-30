function [] = aug(immagine,path,class,n)
%aug prende in input un'immagine, il suo path, la classe di appartenenza e
%il numero di augmented images che deve creare la funzione. La funzione
%salva le immagini modificate in una cartella apposita

x=10; %Valore necessario per non far sovrascrivere le immagini modificate

s = 'C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\F A\';

for i = 1:n
    
    %Augmentation
    tform = randomAffine2d('Rotation',[-90 90], 'XReflection', true, 'YReflection', true,...
        'YTranslation', [-10 10], 'XTranslation',[-10 10]);
    immagine_w = imwarp(immagine,tform);
    
    %Saving
    imwrite(immagine_w,[s class '\' path(end-20:end-4) int2str(x) class '.png']);
    
    %Incrementa numero copia
    x=x+1;

end

