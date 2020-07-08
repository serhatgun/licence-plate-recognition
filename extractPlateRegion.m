function [th_im] = extractPlateRegion(eq_im,bin_im,show)

edges = edge(bin_im, 'canny');
filled = imfill(edges,'holes');
%structure = strel('rectangle',[2 5]);
%closed_im = imclose(fim,structure);
extracted_im = bwareaopen(filled,3000);

figure, imshow(extracted_im)

[rows, columns] = find(extracted_im);
row1 = min(rows);
row2 = max(rows);
col1 = min(columns);
col2 = max(columns);

extracted_binim = extracted_im(row1:row2, col1:col2); % Crop image.
cropped_im = eq_im(row1:row2, col1:col2);
extracted_im = immultiply(cropped_im, extracted_binim);

figure, imshow(extracted_im)

%Postprocessing
filter.filter_type = 'median';
filter.filter_size = 3;
filter.params.form = '';
filter.params.degree = 1;

filtered_im = my_filters(extracted_im,filter);
%filtered_im = locallapfilt(extracted_im,0.2,0.5);

level = graythresh(filtered_im);
th_im = imbinarize(filtered_im,level);
th_im = imcomplement(th_im);

if show
    figure
    subplot(1,2,1), imshow(edges)
    title('Edged Image(canny)')
    subplot(1,2,2), imshow(filled)
    title('Filled Image')
end
end