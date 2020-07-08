function [characters] = extractCharacters(extracted_plate,charSize,show)

[L,n] = bwlabel(extracted_plate);
blobs = regionprops(L,'BoundingBox');

j=1;
for i=1:n
    if(blobs(i).BoundingBox(3)*blobs(i).BoundingBox(4) < 5000 && blobs(i).BoundingBox(3)*blobs(i).BoundingBox(4) > 250)
        subImage = imcrop(extracted_plate, blobs(i).BoundingBox);
        subImage = imcomplement(subImage);
        characters(:,:,j) = imresize(subImage,charSize,'bicubic');
        
        SE = strel("cube",1);
        characters(:,:,j) = imdilate(characters(:,:,j),SE);
        j = j+1;
    end
end

if show
    figure
    imshow(extracted_plate)
    for i=1:n
        rectangle('Position',blobs(i).BoundingBox, 'Edgecolor', 'g');
    end
    title('Bounding Boxes')
    
    figure
    for i = 1:size(characters,3)
        imwrite(characters(:,:,i),string(i)+'.png');
        subplot(1,size(characters,3),i)
        imshow(characters(:,:,i))
    end
    title('Extracted Characters')
end
end