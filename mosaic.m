% Shane Davis
% sh524414
% CAP6419 - 3D Computer Vision
% Assignment 3
% Image mosiac generator from a sequence of images
% Make sure to disable auto-focus on images

RANSAC_THRESHOLD = 0.001;

% Load the images from the directory into a cell list
imgs_dir = uigetdir(title="Select the directory with the images to " + ...
    "create the mosaic from.");
img_names = dir(fullfile(imgs_dir, '*.jpg'));
src_imgs = cell(length(img_names), 1);
for i = 1 : length(img_names)
    % Name should be in the format name_n.jpg (where n is the seq number)
    img_name = img_names(i).name;
    img_num = split(img_name, "_");
    img_prefix = string(img_num(1));
    img_num = split(img_num(2), ".");
    img_num = str2double(string(img_num(1)));
    img_path = fullfile(img_names(i).folder, img_name);
    src_imgs{img_num} = imread(img_path);
end

% Determine middle image to use
middle_index = floor(length(src_imgs) / 2);

% Create a large enough canvas to fit all the images onto it
[middle_height, middle_width] = size(src_imgs{middle_index}, 1:2);
canvas_height = middle_height * 2;
canvas_width = middle_width * 4;
canvas = uint8(zeros(canvas_height, canvas_width, 3));

% Place the middle image from the sequence onto the center of the canvas
canvas_center = [floor(canvas_height/2), floor(canvas_width/2)];
middle_center = [floor(middle_height/2), floor(middle_width/2)];
starting_middle_pos = canvas_center - middle_center;
ending_middle_pos = starting_middle_pos + [middle_height, middle_width] - 1;
canvas(starting_middle_pos(1):ending_middle_pos(1), ...
       starting_middle_pos(2):ending_middle_pos(2), :) = src_imgs{middle_index};
min_valid_y = starting_middle_pos(1);
max_valid_y = ending_middle_pos(1);
min_valid_x = starting_middle_pos(2);
max_valid_x = ending_middle_pos(2);

% Pre-compute middle image related data once
middle_img = src_imgs{middle_index};
middle_gray_img = rgb2gray(middle_img);
middle_sift_points = detectSIFTFeatures(middle_gray_img);
[middle_features, middle_valid_points] = extractFeatures(middle_gray_img, middle_sift_points);
i = middle_index - 1;
left_done = false;
while i <= length(src_imgs)
    chosen_img = src_imgs{i};
    [chosen_height, chosen_width] = size(src_imgs{i}, 1:2);
    chosen_gray_img = rgb2gray(chosen_img);
    
    % Algorithm 4.6 from textbook (Page 123)

    % (i) Interest points
    chosen_sift_points = detectSIFTFeatures(chosen_gray_img);
    [chosen_features, chosen_valid_points] = extractFeatures(chosen_gray_img, chosen_sift_points);
    
    % (ii) Putative correspondences
    index_pairs = matchFeatures(middle_features, chosen_features);
    
    % Get the locations of the corresponding points from both images
    matched_points_middle = middle_valid_points(index_pairs(:, 1), :);
    matched_points_chosen = chosen_valid_points(index_pairs(:, 2), :);
    middle_locations = matched_points_middle.Location + [starting_middle_pos(2), starting_middle_pos(1)];
    chosen_locations = matched_points_chosen.Location;

    % Convert locations to projective coordinates and normalise them
    [middle_X, middle_T] = normalise2dpts([middle_locations(:, 1)'; ...
                                           middle_locations(:, 2)'; ...
                                           ones(1, length(middle_locations))]);
    [chosen_X, chosen_T] = normalise2dpts([chosen_locations(:, 1)'; ...
                                           chosen_locations(:, 2)'; ...
                                           ones(1, length(chosen_locations))]);
    
    % (iii-v) RANSAC robust estimation, Optimal estimation, Guided matching
    % Use Peter's ransac function to determine the inliers
    [~, inliers] = ransac([chosen_X; middle_X], ...
        @norm_DLT, @homogeneous_distance_threshold_2d, @degenerate_check, ...
        4, RANSAC_THRESHOLD);
    num_inliers = size(inliers, 2);
    if num_inliers < 10
        disp("Skipping image number=" + i + ", # of inliers=" + num_inliers);
    else
        % Pick out all the inliers from the RANSAC and calculate the
        % homography using only those points & denormalise it
        H = middle_T^(-1) * norm_DLT(chosen_X(:,inliers), middle_X(:,inliers)) * chosen_T;
    
        % Place the chosen image on a temporary canvas and apply the homography
        temp_canvas = uint8(zeros(canvas_height, canvas_width, 3));
        temp_canvas(1:chosen_height, 1:chosen_width, :) = chosen_img;
        tform = projective2d(H');
        temp_canvas = imwarp(temp_canvas, tform, OutputView=imref2d(size(temp_canvas)));
            
        % Place the image onto the canvas
        % (I couldn't figure out a way to avoid the for loop easily)
        [x_limits, y_limits] = outputLimits(tform, [1 chosen_width], [1 chosen_height]);
        x_limits = [max(1, floor(x_limits(1))), min(canvas_width, ceil(x_limits(2)))];
        y_limits = [max(1, floor(y_limits(1))), min(canvas_height, ceil(y_limits(2)))];
        for y = y_limits(1) : y_limits(2)
            for x = x_limits(1) : x_limits(2)
                if isequal(reshape(canvas(y, x, :), [3, 1]), [0; 0; 0])
                    canvas(y, x, :) = temp_canvas(y, x, :);

                    % Update the min/max positions
                    if x < min_valid_x
                        min_valid_x = x;
                    elseif x > max_valid_x
                        max_valid_x = x;
                    end

                    if y < min_valid_y
                        min_valid_y = y;
                    elseif y > max_valid_y
                        max_valid_y = y;
                    end
                end
            end
        end
    end

    % Iterate to the next image in the sequence
    if i == 1
        i = middle_index;
        left_done = true;
    end

    if ~left_done
        i = i - 1;
    else
        i = i + 1;
    end
end
canvas_crop = canvas(min_valid_y:max_valid_y, min_valid_x:max_valid_x, :);
imshow(canvas_crop);
imwrite(canvas_crop, strcat(img_prefix, "_mosaic.jpg"));

% RANSAC Homography Helper Functions
function H = norm_DLT(X1, X2)
    % Creates a homography where all points in X1 are mapped to X2
    % Algorithm 4.2 from textbook (Page 109)
    % X1 = x', X2 = x

    % RANSAC Compatibility fix:
    if isequal(size(X1), [6, 4])
        X = X1;
        X1 = X(1:3,:);
	    X2 = X(4:6,:);
    end

    % (i) Normalization of X1
    [X1_norm, T1] = normalise2dpts(X1);

    % (ii) Normalization of X2
    [X2_norm, T2] = normalise2dpts(X2);

    % (iii) DLT (Algorithm 4.1 from textbook (Page 91))
    A = zeros(3 * length(X1), 9); 
    z = zeros(3, 1);
    for point = 1 : length(X1)
        % Loop through the points and stack onto A matrix
        % When referencing the 4.1 equation from the book,
        % X2_norm=x_i, X1_norm=x'_i=(x_prime_i, y_prime_i,
        % w_prime_i)
        x_i = X1_norm(:, point);
        x_prime_i = X2_norm(1, point);
        y_prime_i = X2_norm(2, point);
        w_prime_i = X2_norm(3, point);

        A((3 * point) - 2, :) = [z', -w_prime_i * x_i', y_prime_i * x_i'];
        A((3 * point) - 1, :) = [w_prime_i * x_i', z', -x_prime_i * x_i'];
        A(3 * point, :) = [-y_prime_i * x_i', x_prime_i * x_i', z'];
    end
    % Extract the solution for the H
    [~, ~, V_transpose] = svd(A);
    % H is the last column of V_transpose, which is 1x9, reshape to 3x3
    H_norm = V_transpose(:, 9);
    H_norm = reshape(H_norm, 3, 3)';

    % (iv) Denormalization
    H = T2^(-1) * H_norm * T1;
end

function is_degen = degenerate_check(X)
    % Checks if 3-pair from 2 sets of 4 points are co-linear, which 
    % would mean that the 4 points are degenerate
    % Based on Peter's code.

    % Break down the X into the two sets of 4 points (x1, x2 each contain 4
    % points)
    x1 = X(1:3, :);
    x2 = X(4:6, :);

    is_degen = 0;
    for i = 1 : 2
        % Check each set for all possible combinations of 3 pairs of points
        if i == 1
            x_i = x1;
        else
            x_i = x2;
        end
        if iscolinear(x_i(:,1), x_i(:,2), x_i(:,3)) || ...
                iscolinear(x_i(:,1), x_i(:,2), x_i(:,4)) || ...
                iscolinear(x_i(:,1), x_i(:,3), x_i(:,4)) || ...
                iscolinear(x_i(:,2), x_i(:,3), x_i(:,4))
            is_degen = 1;
        end
    end
end

function [inliers, H] = homogeneous_distance_threshold_2d(H, X, threshold)
    % Finds if the points in X are within a threshold distance after
    % applying the homography H
    % Based on Peter's code, with additional explanations.
    
    % Grab the two sets of points from X and normalise them
    X1 = hnormalise(X(1:3,:)); % Current position
    X2 = hnormalise(X(4:6,:)); % New position (found position, not calculated)   
    
    % Apply the homography to the two points and normalise them
    % Move X1 to where it will be after the homography is applied. Ideally,
    % transformed_X1 = X2 since that's where we want the point to be at.
    transformed_X1 = hnormalise(H * X1);  
    % Move X2 back to where it should be before the homography would've 
    % been applied. Ideally, transformed_X2 = X1 since that's where we want
    % the point to be at
    transformed_X2 = hnormalise(H^(-1) * X2);  
    
    % Calculate the distance error that was made from applying the
    % homography.
    total_distance = sum((X1-transformed_X2).^2) + ...
                     sum((X2-transformed_X1).^2);
    inliers = find(abs(total_distance) < threshold);  
end
