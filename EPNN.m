function L2_ACC = EPNN(x,y,x1,y1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%	(x,y) - training, x: Nx*dim, y: 1...Nc
%%% (x1,y1) - test, x1: Nx*dim, y1: 1...Nc
%%% L2_ACC - accuracy results
%%%
%%% prerequisite:
%%%     VLFEAT is needed to be installed first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cls_num = max(y);
order = randperm(length(y))-1;

%%%
L2_ACC = [];
sample_num = 60;    %   this parameter should be tuned based on your applications
iter_num = 50;      %   this parameter should be tuned based on your applications
for si = 1:length(sample_num)
    kdlabels = cell(iter_num,1);
    kdforest = cell(iter_num,1);
    kdX = cell(iter_num,1);
	L2_acc = [];
    for iter = 1:iter_num
        tix = [];
        for i = 1:cls_num
            ind = find(y==i);
            ix = randsample(length(ind),min(length(ind),sample_num(si)));
            tix = [tix; ind(ix)];
        end
        tx = x(tix,:);
        ty = y(tix);

        %%% model
        B = mexLMP_RNBNN(single(x'),int32(y-1),int32(order),single(tx'),int32(ty-1),0.1,1e5);  
        kdX{iter} = B;
        kdtree = vl_kdtreebuild(B) ;        
        kdforest{iter} = kdtree;
        kdlabels{iter} = ty;
        
        L2_model = LearnClassModels_liblinear(x,y,kdX,kdforest,kdlabels,cls_num,iter);     
       

        %%% kdtree query
        I = repmat([1:size(x1,1)]',[1 iter]);
        J = [];
        S = [];
        for i = 1:iter
            [index, distance] = vl_kdtreequery(kdforest{i}, kdX{i}, single(x1)') ;
            Y = kdlabels{i}(index);    
            J = [J; cls_num*(i-1)+Y];
            S = [S, distance'];
        end
        S = normalize(S,2);
        X = sparse(double(I(:)),double(J),double(S(:)),size(x1,1),cls_num*iter);

        [pred_label acc pred_val] = predict(double(y1),sparse(double(X)),L2_model);
        L2_acc = [L2_acc; acc];
    end
    L2_ACC = [L2_ACC, L2_acc];
end

return;




