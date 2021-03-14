img=imread('d.jpg');
B=rgb2gray(img);
subplot(2,2,1)
imshow(B);
title('original image')
I=double(B);
for i=1:size(I,1)-2
for j=1:size(I,2)-2
mx=((2*I(i+2,j+1)+I(i+2,j)+I(i+2,j+2))-(2*I(i,j+1)+I(i,j)+I(i,j+2)));
my=((2*I(i+1,j+2)+I(i,j+2)+I(i+2,j+2))-(2*I(i+1,j)+I(i,j)+I(i+2,j)));
B(i,j)=sqrt(mx.^2+my.^2);
end
end
Thresh=100;
B=max(B,Thresh);
B(B==round(Thresh))=0;
B=uint8(B);
subplot(2,2,2)
imshow(B);title('normalization');
clear all;
close all;
clc;
%Reading the image
Img=imread('d.jpg');
%%Pre Processing and Normalisation
figure;imshow(Img);title('INPUT EYE IMAGE');
%%Step 1: Segmentation
Gray_imag=rgb2gray(Img);
figure;imshow(Gray_imag);title('Segmentation');
%Deleting extra portion
t2=Gray_imag(65:708);
t3=t2(18:563);
figure;
imshow(t3);
title('IMAGE after Deleting extra portion');
%%Step 2: Resizing the image(546x644) to 512 x 512
t4=imresize(t3,[512,512],'bilinear');
figure;imshow(t4);
title('IMAGE after resize');
%%Step 3: Histogram Equlisation
Hist_eq_img = histeq(t4,512);
figure;imshow(Hist_eq_img);
title('IMAGE after Histogram Equlisation');
% Step 4: Gaussian Filtering
G = fspecial('gaussian',[512 512],20);
%Filter it
Hist_eq_img=double(Hist_eq_img);
Ig = imfilter(Hist_eq_img,G,'same');
%Display
%%Step 5: Canny Edge detection
BW2 = edge(Ig,'canny',0.53,1);
figure;imshow(BW2);title('Segmentation');
%Read the original RGB input image
[X1,map1]=imread('d.jpg');
subplot(2,2,1), imshow(X1,map1);title('original image')
%convert it to gray scale
[X2,map2]=imread('d.jpg');
B=rgb2gray(X2);
%resize the image to 160x160 pixels
image_resize=imresize(B, [160 260]);
%apply im2double
image_resize=im2double(image_resize);
%show the image
%Gabor filter size 7x7 and orientation 90 degree
%declare the variables
gamma=0.3; %aspect ratio
psi=0; %phase
theta=90; %orientation
bw=2.8; %bandwidth or effective width
lambda=3.5; % wavelength
pi=180;
for x=1:160
for y=1:260
x_theta=image_resize(x,y)*cos(theta)+image_resize(x,y)*sin(theta);
y_theta=-image_resize(x,y)*sin(theta)+image_resize(x,y)*cos(theta);
gb(x,y)= exp(-(x_theta.^2/2*bw^2+ gamma^2*y_theta.^2/2*bw^2))*cos(2*pi/lambda*x_theta+psi);
end
end
subplot(2,2,2), imshow(gb,map2);title('gaberfilter')
imdata = imread('d.jpg');
D = pdist2(241,14,'hamming');
% reading the image
img=imread('d.jpg');
B=rgb2gray(img);
subplot(2,2,1)
%display the image
imshow(B);title('original image')
pause(2)
I=double(B);
%fixing treshold value before normalization
Thresh=100;
B=max(B,Thresh);
subplot(2,2,2)
imshow(B);title('after fixing treshold value before normalization');
img=imread('d.jpg');
B=rgb2gray(img);
subplot(2,2,1)
imshow(B);
title('original image')
pause(2)
I=double(B);
for i=1:size(I,1)-2
for j=1:size(I,2)-2
mx=((2*I(i+2,j+1)+I(i+2,j)+I(i+2,j+2))-(2*I(i,j+1)+I(i,j)+I(i,j+2)));
my=((2*I(i+1,j+2)+I(i,j+2)+I(i+2,j+2))-(2*I(i+1,j)+I(i,j)+I(i+2,j)));
B(i,j)=sqrt(mx.^2+my.^2);
end
end
B=imread('d.jpg');
subplot(2,2,1)
imshow(B);title('original image')
I=double(B)
for i=1:size(I,1)-2
for j=1:size(I,2)-2
mx=((2*I(i+2,j+1)+I(i+2,j)+I(i+2,j+2))-(2*I(i,j+1)+I(i,j)+I(i,j+2)));
my=((2*I(i+1,j+2)+I(i,j+2)+I(i+2,j+2))-(2*I(i+1,j)+I(i,j)+I(i+2,j)));
B(i,j)=sqrt(mx.^2+my.^2);
end
end
subplot(2,2,2)
imshow(B); title('PUPIL DETECTION');
pause(2)
A = imread('d.jpg');
B = imread('d.jpg');
D = im2double(A);
E = im2double(B);
G = rgb2gray(D);
H= rgb2gray(E);
hn1 = imhist(G);
hn2 = imhist(H);
subplot(2,2,1);subimage(A)
subplot(2,2,2);subimage(B)
f = sum((hn1 - hn2).^2);
disp(f)
if(f==0)
disp('iris matched ');
else if(f~=0)
disp('iris not matched');
end
end
A=imread('d.jpg');  
H= rgb2gray(A);  
subplot(2,2,1);
imshow(A);  
title('iris');  
F=fftn(H);  
subplot(2,2,2); 
imshow(F);  
title('code generated'); 
figure,subplot(2,2,1);
imshow(F);  
title('iris');
F1=ifftn(F); 
subplot(2,2,2); 
imshow(A); 
title('code generated'); 