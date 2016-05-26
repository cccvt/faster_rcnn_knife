function script_faster_rcnn_knife_DR()
close all;
clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.per_nms_topN           = 6000;
opts.nms_overlap_thres      = 0.7;
opts.after_nms_topN         = 300;
opts.use_gpu                = true;

opts.test_scales            = 600;

%% --new config
% do validation, or not 
opts.do_val                 = true; 

% test dataset DR
dataset_DR                  = [];
dataset_DR                  = bag_DR_FPR_imdb(dataset_DR, 'DR', false);

fprintf('\n***************\nDR test\n***************\n');

%load('knife_model.mat');
%load('knife_config.mat');

output_dir = './output/faster_rcnn_final/faster_rcnn_VOC2007_ZF';
load(fullfile(output_dir, 'model'));

%% -- setup rpn_net and fast_rcnn_net
rpn_net.test_net_def_file = fullfile(output_dir, proposal_detection_model.proposal_net_def);
rpn_net.nms.per_nms_topN = -1;
rpn_net.nms.nms_overlap_thres = 0.7;
rpn_net.nms.after_nms_topN = 2000;
rpn_net.cache_name = 'test_nocache_rpn';
rpn_net.output_model_file = fullfile(output_dir, proposal_detection_model.proposal_net);

fast_rcnn_net.test_net_def_file = fullfile(output_dir, proposal_detection_model.detection_net_def);
fast_rcnn_net.cache_name = 'test_nocache_fast_rcnn';
fast_rcnn_net.output_model_file = fullfile(output_dir, proposal_detection_model.detection_net);

%% -- test on test set
dataset_DR.roidb_test       = Faster_RCNN_Train.do_proposal_test(proposal_detection_model.conf_proposal, rpn_net, dataset_DR.imdb_test, dataset_DR.roidb_test);
opts.final_mAP              = Faster_RCNN_Train.do_fast_rcnn_test(proposal_detection_model.conf_detection, fast_rcnn_net, dataset_DR.imdb_test, dataset_DR.roidb_test);

caffe.reset_all(); 
clear mex;

end
