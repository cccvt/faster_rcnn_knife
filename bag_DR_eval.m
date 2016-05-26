function DR = bag_DR_eval(VOCopts,id,cls,draw)

% load test set
[gtids,t]=textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s %d');

% load ground truth objects
tic;
npos=0;
gt(length(gtids))=struct('BB',[],'diff',[],'det',[]);
for i=1:length(gtids)
    % display progress
    if toc>1
        fprintf('%s: pr: load: %d/%d\n',cls,i,length(gtids));
        drawnow;
        tic;
    end
    
    % read annotation
    rec=PASreadrecord(sprintf(VOCopts.annopath,gtids{i}));
    
    % extract objects of class
    clsinds=strmatch(cls,{rec.objects(:).class},'exact');
    gt(i).BB=cat(1,rec.objects(clsinds).bbox)';
    gt(i).diff=[rec.objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
end

% load results
[ids,confidence,b1,b2,b3,b4]=textread(sprintf(VOCopts.detrespath,id,cls),'%s %f %f %f %f %f');
BB=[b1 b2 b3 b4]';

% sort detections by decreasing confidence
[sc,si]=sort(-confidence);
ids=ids(si);
conf_dec=confidence(si);
BB=BB(:,si);

% assign detections to ground truth objects
nd=length(confidence);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;

DR=zeros(nd,2);
% DR(i,1): detection rate
% DR(i,2): confidence value
is_detected=zeros(length(gtids));
num_detected=0;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',cls,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=strmatch(ids{d},gtids,'exact');
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end

    % assign detection to ground truth object if any
    bb=BB(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).BB,2)
        bbgt=gt(i).BB(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 && ih>0                
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
               (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
               iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    DR(d,2)=conf_dec(d);
    % assign detection as true positive/don't care/false positive
    if ovmax>=VOCopts.minoverlap
        if is_detected(i)==0
            is_detected(i)=1;
            num_detected=num_detected+1;
        end
    end
    DR(d,1)=num_detected/npos;
end

if draw
    % plot precision/recall
    plot(DR(:,2),DR(:,1),'-');
    grid;
    xlabel 'confidence'
    ylabel 'detection rate'
end

%save rate-conf data
DR_conf=[DR(:,2) DR(:,1)];
save('DR_conf.mat','DR_conf');

end

