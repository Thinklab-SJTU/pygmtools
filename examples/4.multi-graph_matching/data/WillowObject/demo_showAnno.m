function demo_showAnno( )
% demo function to visualize the dataset images and their annotations
% by Minsu Cho, Inria - WILLOW / ENS 

classes = {'Face', 'Motorbike', 'Car', 'Duck', 'Winebottle'};
% setting
conf.dataDir = '.';
conf.imgDir=[conf.dataDir '/WILLOW-ObjectClass'];
conf.annoDir=[conf.dataDir '/WILLOW-ObjectClass'];

conf.class = classes{5};           % change here to see the other classes

disp(['show images and their annotations in class ' conf.class '...']);
listOfFile = dir(fullfile(conf.imgDir, conf.class, '*.png'));

colorCode = makeColorCode(100);
hFig = figure('Name', 'image & annotation', 'NumberTitle', 'off'); 
iEnd = numel(listOfFile);
for i=1:iEnd
        filePathName = fullfile(conf.imgDir, conf.class, listOfFile(i).name );
        anno_filePathName = fullfile(conf.annoDir, conf.class, [ listOfFile(i).name(1:end-4) '.mat' ]);
        
        img = imread(filePathName);
        pts_coord = []; 
        if exist(anno_filePathName)
            load(anno_filePathName, 'pts_coord');
        end
        
        figure(hFig); clf;  imshow(rgb2gray(img)); hold on;
        for j=1:size(pts_coord,2)
            plot(pts_coord(1,j),pts_coord(2,j),'o','MarkerEdgeColor','k',...
                'MarkerFaceColor',colorCode(:,j),'MarkerSize', 10);
        end
        title(sprintf('%s %4d/%4d : %s',conf.class,i,iEnd,listOfFile(i).name));
        
        pause;
end

end

function [ priorColorCode ] = makeColorCode( nCol )

priorColorCode(1,:) = [ 1 0 0 ]; 
priorColorCode(2,:) = [ 0 1 0 ]; 
priorColorCode(3,:) = [ 0 0 1 ]; 
priorColorCode(4,:) = [ 0 1 1 ]; 
priorColorCode(5,:) = [ 1 0 1 ]; 
priorColorCode(6,:) = [ 1 1 0 ]; 
priorColorCode(7,:) = [ 1 0.5 0 ]; 
priorColorCode(8,:) = [ 1 0 0.5 ]; 
priorColorCode(9,:) = [ 1 0.5 0.5 ]; 
priorColorCode(10,:) = [ 0.5 1 0 ]; 
priorColorCode(11,:) = [ 0 1 0.5 ]; 
priorColorCode(12,:) = [ 0.5 1 0.5 ]; 

nMore = nCol - size(priorColorCode,1);
if nMore > 0 
    priorColorCode(size(priorColorCode,1)+1:nCol,:) = rand(nMore, 3);
end

priorColorCode = priorColorCode';

end

