% Written by: Harshal Kaushik and Dr. Farzad Yousefian.
% First created: Jan 13th, 2018.
% Latest modified: Jan 28th, 2018.
% Objective: Generate a dataset ('A', and 'b') from the cameraman's
% image. 

function [A, b, n] = data_set(bandw)

% imageName='cameraman_mini.jpg';
imageName = 'cameraman.pgm';
X=imread(imageName);
%figure, imshow(X), title('Original Image')
[l, ~]=size(X);
%convert matrix into vector
x=double(X(:)); %original image vector
[n,~]=size(x);

%motionblur matrix using block Toeplitz matrix
A = mblur(l,bandw,'x');
% A = mblur_light(l,2,'x');

%load ('SimPara.mat');
%multiplication
E = randn(n,1);
E = E / norm(E,'fro');

% b=A*sparse(x)+10*randn(n,1); %b is sparse vector
b = A*sparse(x);
b = b + 0.1*norm(b,'fro')*E;

B=reshape(b,[l l]); %B is sparse matrix
B=full(B); %B is double
B=uint8(B); % B is uint8 for imshow()
F1 = figure; 
imshow(B), 
% title('Blurred Image');
set(gcf,'PaperPositionMode','auto')
saveas(F1, 'BlurredImage.png');

% B = double(B(:));

end
