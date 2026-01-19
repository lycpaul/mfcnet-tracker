%% Create pose keypoint segmentation maps 

% DATA_DIR = "../../../JIGSAWS_dataset/annotations_val/";
DATA_DIR = "/rhf/bg40/Rice-Methodist-Surgical-Skills-Assessement-Data-2025/annotated_dataset_for_toolpose/val/";
NUM_DATA_FOLDERS = 6; 
NUM_SAMPLES_PER_FOLDER = 75; 
POSE_DATA_FNAME = "jaw_poses_left_right.mat"; 

KEYPOINT_SZ = 8; %4; 

IMG_H = 1080; %480; 
IMG_W = 1920; %640; 

%figure; 
for i=1:1:6
    SAVE_DIR = DATA_DIR + "video_" + num2str(i) + "/pose_maps/"; 
    mkdir(SAVE_DIR); 
    string = DATA_DIR + "video_" + num2str(i) + "/" + POSE_DATA_FNAME; 
    load(string); 
    changeFilePaths(gTruth, {["E:\Bhargav_LabPC_data_backup\", "/rhf/bg40/"]});
    % Read pose data frame-by-frame; generate keypoint segmentation maps
    for j=1:NUM_SAMPLES_PER_FOLDER
        IMG_H = size(imread(gTruth.DataSource.Source{j,1}),1);
        IMG_W = size(imread(gTruth.DataSource.Source{j,1}),2); 
        [X, Y] = meshgrid(1:IMG_W, 1:IMG_H); 
        left_instrument_coords = gTruth.LabelData.JawPoseLeft{j,1}; 
        right_instrument_coords = gTruth.LabelData.JawPoseRight{j,1}; 
        pmap_left_tip1 = zeros(IMG_H, IMG_W); 
        pmap_left_base = zeros(IMG_H, IMG_W); 
        pmap_left_tip2 = zeros(IMG_H, IMG_W); 
        if ~isempty(left_instrument_coords)
            lc1x = left_instrument_coords{1,1}(1,1); 
            lc1y = left_instrument_coords{1,1}(1,2); 
            lc2x = left_instrument_coords{1,1}(2,1); 
            lc2y = left_instrument_coords{1,1}(2,2);
            if size(left_instrument_coords{1,1},1)==4
                pmap_left_base(sqrt((X-lc1x).^2 + (Y-lc1y).^2)<=(KEYPOINT_SZ)) = 1;
            else
                if size(left_instrument_coords{1,1},1)<3 
                    lc3x = left_instrument_coords{1,1}(1,1); 
                    lc3y = left_instrument_coords{1,1}(1,2);
                else
                    lc3x = left_instrument_coords{1,1}(3,1); 
                    lc3y = left_instrument_coords{1,1}(3,2); 
                end
                pmap_left_tip1(sqrt((X-lc1x).^2 + (Y-lc1y).^2)<=(KEYPOINT_SZ)) = 1;
                pmap_left_base(sqrt((X-lc2x).^2 + (Y-lc2y).^2)<=(KEYPOINT_SZ)) = 1;
                pmap_left_tip2(sqrt((X-lc3x).^2 + (Y-lc3y).^2)<=(KEYPOINT_SZ)) = 1;
            end
        end
        pmap_left = cat(3, pmap_left_tip1, pmap_left_base, pmap_left_tip2);
        pmap_right_tip1 = zeros(IMG_H, IMG_W);
        pmap_right_base = zeros(IMG_H, IMG_W); 
        pmap_right_tip2 = zeros(IMG_H, IMG_W); 
        if ~isempty(right_instrument_coords)
            rc1x = right_instrument_coords{1,1}(1,1); 
            rc1y = right_instrument_coords{1,1}(1,2); 
            rc2x = right_instrument_coords{1,1}(2,1); 
            rc2y = right_instrument_coords{1,1}(2,2);
            if size(right_instrument_coords{1,1},1)==4
                pmap_right_base(sqrt((X-rc1x).^2 + (Y-rc1y).^2)<=(KEYPOINT_SZ)) = 1;
            else
                if size(right_instrument_coords{1,1},1) < 3
                    rc3x = right_instrument_coords{1,1}(1,1); 
                    rc3y = right_instrument_coords{1,1}(1,2);
                else
                    rc3x = right_instrument_coords{1,1}(3,1); 
                    rc3y = right_instrument_coords{1,1}(3,2);
                end
                pmap_right_tip1(sqrt((X-rc1x).^2 + (Y-rc1y).^2)<=(KEYPOINT_SZ)) = 1; 
                pmap_right_base(sqrt((X-rc2x).^2 + (Y-rc2y).^2)<=(KEYPOINT_SZ)) = 1; 
                pmap_right_tip2(sqrt((X-rc3x).^2 + (Y-rc3y).^2)<=(KEYPOINT_SZ)) = 1;
            end
        end
        pmap_right = cat(3, pmap_right_tip1, pmap_right_base, pmap_right_tip2);
        imwrite(pmap_left, SAVE_DIR+"framel"+pad(num2str(j-1),3,'left','0')+'.png');
        imwrite(pmap_right, SAVE_DIR+"framer"+pad(num2str(j-1),3,'left','0')+'.png'); 
        %img = imread(gTruth.DataSource.Source{j,1}); 
        %imshow(img); hold on; h = imshow(pmap_left+pmap_right); set(h,'AlphaData',0.5);hold off;pause;
    end
    disp(i);
end
