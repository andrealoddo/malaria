function [] = delete_aug_images(char)
%delete_aug_images elimina le augmented images salvate con la funzione aug.

%Cartella da svuotare
myFolder = ['C:\Users\Corrado\Desktop\Transfer Learning for Malaria Diagnosis\F A\' char];
filePattern = fullfile(myFolder, '*.png'); 
theFiles = dir(filePattern);

%Eliminazione di tutte le immagini all'interno
for k = 1 : length(theFiles)
  baseFileName = theFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  delete(fullFileName);
end

end

