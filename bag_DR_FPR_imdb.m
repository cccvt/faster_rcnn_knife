function dataset = bag_DR_FPR_imdb(dataset, usage, use_flip)
% Pascal voc 2007 trainval set
% set opts.imdb_train opts.roidb_train 
% or set opts.imdb_test opts.roidb_train

% change to point to your devkit install
devkit = './datasets/knife_0519';

switch usage
    case {'DR'}
        dataset.imdb_test     = imdb_from_voc(devkit, 'test_DR', '2007', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    case {'FPR'}
        dataset.imdb_test     = imdb_from_voc(devkit, 'test_FPR', '2007', use_flip) ;
        dataset.roidb_test    = dataset.imdb_test.roidb_func(dataset.imdb_test);
    otherwise
        error('usage = ''DR'' or ''FPR''');
end

end