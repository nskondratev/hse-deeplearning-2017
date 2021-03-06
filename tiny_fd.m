function boxes = tiny_fd(image_path)
if nargin < 3 
  prob_thresh = 0.5;
end
if nargin < 4 
  nms_thresh = 0.1;
end
if nargin < 5 
  gpu_id = 0;  % 0 means no use of GPU (matconvnet starts with 1) 
end

addpath tiny/matconvnet;
addpath tiny/matconvnet/matlab;
vl_setupnn;

addpath tiny/utils;
addpath tiny/toolbox/nms;
addpath tiny/toolbox/export_fig;

%
MAX_INPUT_DIM = 5000;
MAX_DISP_DIM = 3000;

% specify pretrained model (download if needed)
model_dir = './tiny/trained_models';
if ~exist(model_dir)
  mkdir(model_dir);
end
model_path = fullfile(model_dir, 'hr_res101.mat');
if ~exist(model_path)
  url = 'https://www.cs.cmu.edu/~peiyunh/tiny/hr_res101.mat';
  cmd = ['wget -O ' model_path ' ' url];
  system(cmd);
end
% loadng pretrained model (and some final touches)
fprintf('Loading pretrained detector model...\n');
net = load(model_path);
net = dagnn.DagNN.loadobj(net.net);
net.mode = 'test';
if gpu_id > 0 % for matconvnet it starts with 1 
  gpuDevice(gpu_id);
  net.move('gpu');
end
net.layers(net.getLayerIndex('score4')).block.crop = [1,2,1,2];
net.addLayer('cropx',dagnn.Crop('crop',[0 0]),...
             {'score_res3', 'score4'}, 'score_res3c'); 
net.setLayerInputs('fusex', {'score_res3c', 'score4'});
net.addLayer('prob_cls', dagnn.Sigmoid(), 'score_cls', 'prob_cls');
averageImage = reshape(net.meta.normalization.averageImage,1,1,3);

% reference boxes of templates
clusters = net.meta.clusters;
clusters_h = clusters(:,4) - clusters(:,2) + 1;
clusters_w = clusters(:,3) - clusters(:,1) + 1;
normal_idx = find(clusters(:,5) == 1);

% by default, we look at three resolutions (.5X, 1X, 2X)
%scales = [-1 0 1]; % update: adapt to image resolution (see below)

% initialize output 
bboxes = [];

% load input
t1 = tic; 
[~,name,ext] = fileparts(image_path);
try
  raw_img = imread(image_path);
catch
  error(sprintf('Invalid input image path: %s', image_path));
  return;
end

% process input at different scales 
raw_img = single(raw_img);
[raw_h, raw_w, ~] = size(raw_img) ;
min_scale = min(floor(log2(max(clusters_w(normal_idx)/raw_w))),...
                floor(log2(max(clusters_h(normal_idx)/raw_h))));
max_scale = min(1, -log2(max(raw_h, raw_w)/MAX_INPUT_DIM));
scales = [min_scale:0, 0.5:0.5:max_scale];

for s = 2.^scales
  img = imresize(raw_img, s, 'bilinear');
  img = bsxfun(@minus, img, averageImage);

  fprintf('Processing %s at scale %f.\n', image_path, s);
  
  if strcmp(net.device, 'gpu')
    img = gpuArray(img);
  end

  % we don't run every template on every scale
  % ids of templates to ignore 
  tids = [];
  if s <= 1, tids = 5:12;
  else, tids = [5:12 19:25];
  end
  ignoredTids = setdiff(1:size(clusters,1), tids);

  % run through the net
  [img_h, img_w, ~] = size(img);
  inputs = {'data', img};
  net.eval(inputs);

  % collect scores 
  score_cls = gather(net.vars(net.getVarIndex('score_cls')).value);
  score_reg = gather(net.vars(net.getVarIndex('score_reg')).value);
  prob_cls = gather(net.vars(net.getVarIndex('prob_cls')).value);
  prob_cls(:,:,ignoredTids) = 0;

  % threshold for detection
  idx = find(prob_cls > prob_thresh);
  [fy,fx,fc] = ind2sub(size(prob_cls), idx);

  % interpret heatmap into bounding boxes 
  cy = (fy-1)*8 - 1; cx = (fx-1)*8 - 1;
  ch = clusters(fc,4) - clusters(fc,2) + 1;
  cw = clusters(fc,3) - clusters(fc,1) + 1;

  % extract bounding box refinement
  Nt = size(clusters, 1); 
  tx = score_reg(:,:,1:Nt); 
  ty = score_reg(:,:,Nt+1:2*Nt); 
  tw = score_reg(:,:,2*Nt+1:3*Nt); 
  th = score_reg(:,:,3*Nt+1:4*Nt); 

  % refine bounding boxes
  dcx = cw .* tx(idx); 
  dcy = ch .* ty(idx);
  rcx = cx + dcx;
  rcy = cy + dcy;
  rcw = cw .* exp(tw(idx));
  rch = ch .* exp(th(idx));

  %
  scores = score_cls(idx);
  tmp_bboxes = [rcx-rcw/2, rcy-rch/2, rcx+rcw/2, rcy+rch/2];

  tmp_bboxes = horzcat(tmp_bboxes ./ s, scores);

  bboxes = vertcat(bboxes, tmp_bboxes);
end

% nms 
ridx = nms(bboxes(:,[1:4 end]), nms_thresh); 
bboxes = bboxes(ridx,:);

%
bboxes(:,[2 4]) = max(1, min(raw_h, bboxes(:,[2 4])));
bboxes(:,[1 3]) = max(1, min(raw_w, bboxes(:,[1 3])));

%
t2 = toc(t1);

boxes = bboxes;
boxes(:,1:4) = boxes(:,1:4) * 1.0;

fprintf('Detection was finished in %f seconds\n', t2);

% free gpu device
if gpu_id > 0 
  gpuDevice([]);
end