function FPR = bag_FPR_eval(imdb, detect_result_file, draw)

% load results
[ids,confidence,b1,b2,b3,b4]=textread(detect_result_file,'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
conf_dec=confidence(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tic;

FPR=zeros(nd,2);
% FPR(i,1): false positive rate
% FPR(i,2): confidence value
nneg=length(imdb.image_ids);
is_detected=zeros(nneg);
num_detected=0;
for d=1:nd
    % display progress
    if toc>1
        fprintf('pr: compute: %d/%d\n',d,nd);
        drawnow;
        tic;
    end
    FPR(d,2)=conf_dec(d);
    i = find(strcmp(imdb.image_ids, ids{d}) > 0);
    if is_detected(i)==0
        is_detected(i)=1;
        num_detected=num_detected+1;
    end
    FPR(d,1)=num_detected/nneg;
end

if draw
    % plot precision/recall
    plot(FPR(:,2),FPR(:,1),'-');
    grid;
    xlabel 'confidence'
    ylabel 'false positive rate'
end

%save rate-conf data
FPR_conf=[FPR(:,2) FPR(:,1)];
save('FPR_conf.mat','FPR_conf');

end
