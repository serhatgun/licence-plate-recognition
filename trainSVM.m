function [classifier, hogFeatureSize] = trainSVM(cellSize, trainImSize)

trainDataPath = 'classifier' + string(trainImSize(1)) + '.mat';

% No need to train again if it is already trained
if isfile(trainDataPath)
    load(trainDataPath);
    
    % Create hog feature matrix by 2x2 cells
    img = readimage(trainingDataSet, 206);
    img = imresize(img,trainImSize,'bicubic');
    
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    
    [hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',cellSize);
    [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
    [hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
    
    % Show the original image
    figure;
    subplot(2,3,1:3); imshow(img);
    
    % Visualize the HOG features
    subplot(2,3,4);
    plot(vis2x2);
    title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
    
    subplot(2,3,5);
    plot(vis4x4);
    title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
    
    subplot(2,3,6);
    plot(vis8x8);
    title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
    
    
    return
else
    %%%%%%%% Load digit training set %%%%%%%%
    % digitsDir   = fullfile(toolboxdir('vision'), 'visiondata','digits','synthetic');
    % digitsTrainingSet = imageDatastore(digitsDir,'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    trainingDataSet = imageDatastore('trainData','IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % Create hog feature matrix by 2x2 cells
    img = readimage(trainingDataSet, 206);
    img = imresize(img,trainImSize,'bicubic');
    if ndims(img) == 3
        img = rgb2gray(img);
    end
    [hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',cellSize);
    [hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
    [hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);
    
    % Show the original image
    figure;
    subplot(2,3,1:3); imshow(img);
    
    % Visualize the HOG features
    subplot(2,3,4);
    plot(vis2x2);
    title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});
    
    subplot(2,3,5);
    plot(vis4x4);
    title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});
    
    subplot(2,3,6);
    plot(vis8x8);
    title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});
    
    % Get hog feature size
    cellSize = cellSize;
    hogFeatureSize = length(hog_2x2);
    
    % Loop over the trainingSet and extract HOG features from each image. A
    % similar procedure will be used to extract features from the testSet.
    numImages = numel(trainingDataSet.Files);
    trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
    
    for i = 1:numImages
        img = readimage(trainingDataSet, i);
        img = imresize(img,trainImSize,'bicubic');
        
        if ndims(img) == 3
            img = rgb2gray(img);
        end
        
        % Apply pre-processing steps
        img = imbinarize(img);
        trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);
    end
    
    % Get labels for each image.
    trainingLabels = trainingDataSet.Labels;
    
    % fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
    classifier = fitcecoc(trainingFeatures, trainingLabels);
    save(trainDataPath)
end
end