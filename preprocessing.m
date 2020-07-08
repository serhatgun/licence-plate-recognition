function [resized_im,gray_im,eq_im,filtered_im,bin_im] = preprocessing(im, show)

resized_im = imresize(im, [1024 768], 'bicubic');
gray_im = rgb2gray(resized_im);

[eq_im] = my_histogram_equalization(gray_im);

filter.filter_type = 'median';
filter.filter_size = 3;
filter.params.form = '';
filter.params.degree = 1;

filtered_im = my_filters(eq_im,filter);
filtered_im = locallapfilt(eq_im,0.2,0.5);

level = graythresh(filtered_im);
bin_im = imbinarize(filtered_im,level);

if show
    figure
    subplot(2,2,1), imshow(resized_im)
    title('Resized Image')
    subplot(2,2,2), imshow(eq_im)
    title('Gray Level')
    subplot(2,2,3), imshow(gray_im)
    title('Histogram Equalized')
    subplot(2,2,4), imshow(bin_im)
    title('Binary Image')
end
end