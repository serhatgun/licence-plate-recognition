function [finalResult, histogram] = my_histogram_equalization(im)

[rows, cols] = size(im);

index = 1:256;
index = index / 255;

pixelNumber = rows*cols;
finalResult = uint8(zeros(rows,cols));
frequency = zeros(256,1);
pdf = zeros(256,1);
cdf = zeros(256,1);
cummlative = zeros(256,1);
out = zeros(256,1);

for im_rows = 1:rows
    for im_cols = 1:cols
        gray_level = im(im_rows, im_cols);
        frequency(gray_level+1) = frequency(gray_level+1) + 1;
        pdf(gray_level+1) = frequency(gray_level+1)/pixelNumber;
    end
end

% figure(1)
% bar(index, pdf)

% finding cdf
sum = 0;
L = 255;

for i = 1:size(pdf)
    sum = sum + frequency(i);
    cummlative(i) = sum;
    cdf(i) = cummlative(i)/pixelNumber;
    out(i) = round(cdf(i)*L);
end


for im_rows = 1:rows
    for im_cols = 1:cols   
        finalResult(im_rows,im_cols) = out(im(im_rows,im_cols)+1);
    end
end

frequency = zeros(256,1);
histogram = zeros(256,1);

for im_rows = 1:rows
    for im_cols = 1:cols
        gray_level = finalResult(im_rows, im_cols);
        frequency(gray_level+1) = frequency(gray_level+1) + 1;
        histogram(gray_level+1) = frequency(gray_level+1)/pixelNumber;
    end
end


% figure(2)
% bar(index, histogram);

end