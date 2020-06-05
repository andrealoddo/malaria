clear
clc

path1 = 'C:\Users\Corrado\Documents\Informatica\. Tesi\preprocessed_cell_images\Parasitized';
path2 = 'C:\Users\Corrado\Documents\Informatica\. Tesi\preprocessed_cell_images\Uninfected';
malaria_ds1 = imageDatastore(path1);
malaria_ds2 = imageDatastore(path2);

preprocess_ds(malaria_ds1);
preprocess_ds(malaria_ds2);
