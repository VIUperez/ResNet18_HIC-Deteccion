%% Carga de las imágenes. path de datos. 
path='---';
img=imageDatastore(path,'IncludeSubfolders',true,'LabelSource','foldernames');
% Determinar la separación de las imágenes
img_eti=countEachLabel(img);
% Número de imágenes
img_num=length(img.Labels);
%% Crear set de entrenamiento y validación de imágenes
[training_set,validation_set,holdout_img] = splitEachLabel(img,75,25,'randomized'); %75 imágenes de cada para entrenamiento y 25 para validación. 
%Aumentador de datos de imagen 
img_Augmenter = imageDataAugmenter('RandRotation',[-10,10],'RandXTranslation',[-3 3],'RandYTranslation',[-3 3]);
outputSize=[224 224 3];
%Aplicación de los cambios a los set de entrenamiento y validación. Al set
%de validación no se le aplica el aumento. Así, se obtienen los sets
%definitivos. 
training_set_aug=augmentedImageDatastore(outputSize,training_set,'ColorPreprocessing','gray2rgb','DataAugmentation',img_Augmenter);
validation_set_aug=augmentedImageDatastore(outputSize,validation_set,'ColorPreprocessing','gray2rgb');
%% Preparar ResNet18
net = resnet18;
lgraph = layerGraph(net);
lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'}); %Elimina las tres últimas capas de la red
Classes_num = numel(categories(training_set.Labels)); 
new_Layers = [fullyConnectedLayer(Classes_num,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10) softmaxLayer('Name','softmax') classificationLayer('Name','classoutput')]; %Determinar las nuevas capas que se van a añadir
lgraph = addLayers(lgraph,new_Layers);  
lgraph = connectLayers(lgraph,'pool5','fc'); 
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]); 
plot(lgraph) %Se plotea una gráfica de la nueva estructura de la red. 
ylim([0,10]);
%% Entrenamiento de la red 
miniBatchSize = 4; 
MaxEpochs=5; 
numIterationsPerEpoch = floor(numel(training_set.Labels)/miniBatchSize); 
%Opciones de entrenamiento
opt = trainingOptions('sgdm','MiniBatchSize',miniBatchSize,'MaxEpochs',MaxEpochs,'InitialLearnRate',1e-4,'Verbose',false,'Plots','training-progress','ValidationData',validation_set_aug,'ValidationFrequency',numIterationsPerEpoch,'ValidationPatience',Inf);
Transfer_Learn = trainNetwork(training_set_aug,lgraph,opt); %Linea para comenzar el entrenamiento de la red ResNet18
%Matriz de confusión
predictedLabelsValidation = classify(Transfer_Learn,validation_set_aug); 
plotconfusion(validation_set.Labels,predictedLabelsValidation);

%% GradCam
net=Transfer_Learn;
classes = net.Layers(end).Classes;
softmaxName = 'softmax';
featureLayerName = 'res5b_relu';
for i=1:length(validation_set.Files)
   h = figure('Units','normalized','Position',[0.05 0.05 0.9 0.8],'Visible','on');
   [img,fileinfo] = readimage(validation_set,i);
   im=img(:,:,[1 1 1]); %Convertir la imagen de escala de grises a RGB
   imResized = imresize(img, [224 224]);
   imResized=imResized(:,:,[1 1 1]); 
   
   imageActivations = activations(net,imResized,featureLayerName);
   
   scores = squeeze(mean(imageActivations,[1 2]));
   fcWeights = net.Layers(end-2).Weights;
   fcBias = net.Layers(end-2).Bias;
   scores = fcWeights*scores + fcBias;
   [~,classIds] = maxk(scores,4); 
   weightVector = shiftdim(fcWeights(classIds(1),:),-1);
   classActivationMap = sum(imageActivations.*weightVector,3);
   scores = exp(scores)/sum(exp(scores));
   maxScores = scores(classIds); labels = classes(classIds);
   [maxScore, maxID] = max(maxScores);
   labels_max = labels(maxID);
   
   CAMshow(im,classActivationMap) %Línea que muestra la imagen junto con el mapa de activación. 
   title("Predicted: "+string(labels_max) + ", " + string(maxScore)+" (Actual: "+ string(validation_set.Labels(i))+")",'FontSize', 18);
   
   drawnow
end