clc
close all;
clear all;

im = imread('testData/0.jpg');

% HOG feature matrix cell size
cellSize = [4 4];
trainImSize = [16 16];

[resized_im,gray_im,eq_im,filtered_im,bin_im] = preprocessing(im,true);

extracted_plate = extractPlateRegion(eq_im,bin_im,true);

characters = extractCharacters(extracted_plate,trainImSize,true);

[classifier, hogFeatureSize] = trainSVM(cellSize,trainImSize);

[predictedLabels] = predictSVM(classifier, characters, hogFeatureSize, cellSize)'

figure
for i = 1:size(characters,3)
    imwrite(characters(:,:,i),string(i)+'.png');
    subplot(1,size(characters,3),i)
    imshow(characters(:,:,i))
end
title('Extracted Characters')


% result = [string(predictedDigitLabels(1:2))'  string(predictedLetterLabels(3:5))'  string(predictedDigitLabels(6:7))'];
% fprintf('Recognized Plate: ')
% fprintf('%s', result)
% fprintf('\n')
