function [predictedLabels] = predictSVM(classifier,characters,hogFeatureSize,cellSize)

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures] = helperExtractHOGFeaturesFromImageSet(characters, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% % Tabulate the results using a confusion matrix.
% confMat = confusionmat(testLabels, predictedLabels);

% helperDisplayConfusionMatrix(confMat)
% title(string(predictedLabels))
end


function [features] = helperExtractHOGFeaturesFromImageSet(chars, hogFeatureSize, cellSize)

features  = zeros(size(chars,3), hogFeatureSize, 'single');

% Process each image and extract features
for j = 1:size(chars,3)
    img = chars(:,:, j);
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end