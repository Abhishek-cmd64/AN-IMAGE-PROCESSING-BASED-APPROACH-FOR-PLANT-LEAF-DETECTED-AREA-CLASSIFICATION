
clc
close all
clear all
% Image Selection from Data Set
[filename,pathname] = uigetfile({'*.*';'*.bmp';'*.tif';'*.gif';'*.png'},'Pick a Disease Affected Leaf');
I = imread([pathname,filename]);
subplot(1,2,1);
imshow(I)
title('Disease Affected Leaf');

I4=imadjust(I, stretchlim(I));
subplot(1,2,2);
imshow(I4);
title('Contrast Enhanced');

% Use of K Means clustering for segmentation
% Convert Image from RGB Color Space to L*a*b* Color Space 
% The L*a*b* space consists of a luminosity layer 'L*', chromaticity-layer 'a*' and 'b*'.
% All of the color information is in the 'a*' and 'b*' layers.
cform = makecform('srgb2lab');

% Apply the colorform
lab_he = applycform(I,cform); %Apply device-independent color space transformation.

% Classify the colors in a*b* colorspace using K means clustering.
% Since the image has 3 colors create 3 clusters.
% Measure the distance using Euclidean Distance Metric.
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);  %[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean','Replicates',3);
% Label every pixel in tha image using results from K means
pixel_labels = reshape(cluster_idx,nrows,ncols);
%figure,imshow(pixel_labels,[]), title('Image Labeled by Cluster Index');

% Create a blank cell array to store the results of clustering
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end
% Display the contents of the clusters
figure
subplot(1,3,1);
imshow(segmented_images{1});
title('Cluster 1'); 
subplot(1,3,2);
imshow(segmented_images{2});
title('Cluster 2');
subplot(1,3,3);
imshow(segmented_images{3});
title('Cluster 3');
%enhancing the output image
x = inputdlg('Enter the cluster no. containing the disease affected leaf part only:');
I=segmented_images{k}
histI=imhist(I);
imhist(I);title('Histogram of original image');
imshow(I);title('cluster 3')
B=double(I);
histI;
[x y]=size(I);

P=histI/sum(histI);
max=0;
sigma(1)=0;
for T=2:255
    P1=sum(P(1:T));
    P2=sum(P(T+1:256));
    
    mu1=sum((0:T-1)'.*P(1:T))/P1;
    mu2=sum((T:255)'.*P(T+1:256))/P2;
    sigma(T)=P1*P2*((mu1-mu2)^2);
    if sigma(T)>max
        max=sigma(T);
        threshold=T-1;
    end 
    
end
%bw=I<=threshold;
for i=1:x
    for j=1:y       
     if B(i,j)<=threshold
       B(i,j)=0;
     else
       B(i,j)=255;   
     end
    end
end
figure;
imshow(B);title('disease area enhanced');figure
h=imhist(B);
plot(h);title('Histogram of output Image');

seg_img = segmented_images{k};

% Convert to grayscale if image is RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end

% Create the Gray Level Cooccurance Matrices (GLCMs)
glcms = graycomatrix(img);

%Evaluate 13 features from the disease affected region only
% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);

% Put the 13 features in an array
feat_disease = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];
%%
% Load All The Features
load('Training_Data.mat')

% Put the test features into variable 'test'
test = feat_disease;
result = multisvm(Train_Feat,Train_Label,test);
%disp(result);

% Visualize Results
if result == 0
    helpdlg(' Alternaria Alternata ');
    disp(' Alternaria Alternata ');
elseif result == 1
    helpdlg(' Anthracnose ');
    disp('Anthracnose');
elseif result == 2
    helpdlg(' Bacterial Blight ');
    disp(' Bacterial Blight ');
elseif result == 3
    helpdlg(' Cercospora Leaf Spot ');
    disp('Cercospora Leaf Spot');
elseif result == 4
    helpdlg(' Healthy Leaf ');
    disp('Healthy Leaf ');
end
% 
% %% Evaluate Accuracy
% load('Accuracy_Data.mat')
% Accuracy_Percent= zeros(200,1);
% for i = 1:5
% data = Train_Feat;
% %groups = ismember(Train_Label,1);
% groups = ismember(Train_Label,0);
% [train,test] = crossvalind('HoldOut',groups);
% cp = classperf(groups);
% svmStruct = svmtrain(data(train,:),groups(train),'showplot',false,'kernel_function','linear');
% classes = svmclassify(svmStruct,data(test,:),'showplot',false);
% classperf(cp,classes,test);
% Accuracy = cp.CorrectRate;
% Accuracy_Percent(i) = Accuracy.*100;
% end
% Max_Accuracy = max(Accuracy_Percent);
% sprintf('Accuracy of Linear Kernel with 500 iterations is: %g%%',Max_Accuracy)
% 
