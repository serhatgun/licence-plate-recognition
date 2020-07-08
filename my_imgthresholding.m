function [new_im] = my_imgthresholding(im, T)
[x y] = size(im);
G1 = uint8(zeros(1));
G2 = uint8(zeros(1));
meanGL = mean(im(:));

if isempty(T)
    th = meanGL;
else
    th = T;
end

T = 0; it = 0;
while abs(T-th) > 0.1
    T = th;
    indG1 = 1; 
    indG2 = 1;
    for row = 1:x
        for col = 1:y
            if double(im(row,col)) > th
                G1(indG1) = im(row,col);
                indG1 = indG1 + 1;
            else
                G2(indG2) = im(row,col);
                indG2 = indG2 + 1;            
            end
        end
    end
    u1 = mean(G1(:));
    u2 = mean(G2(:));
    th = (u1+u2)/2;
    it = it + 1;
    G1 = uint8(zeros(1));
    G2 = uint8(zeros(1));
end
fprintf('Number of iterations: %d \n', it);

new_im = uint8(zeros(x,y));

for row = 1:x
    for col = 1:y
        if im(row,col) <= uint8(round(th))
            new_im(row,col) = uint8(0);
        else
            new_im(row,col) = uint8(1);
        end
    end
end
end