function L2_model = LearnClassModels_liblinear(x,y,kdX,kdforest,kdlabels,Nc,Nl)
%%%
xlen = size(x,1);

%%% kdtree query
I = repmat([1:xlen]',[1 Nl]);
J = [];
S = [];
for i = 1:Nl
    [index, distance] = vl_kdtreequery(kdforest{i}, kdX{i}, single(x)') ;
    Y = kdlabels{i}(index);    
    J = [J; Nc*(i-1)+Y];
    S = [S, distance'];
end
S = normalize(S,2);
X = sparse(double(I(:)),double(J(:)),double(S(:)),xlen,Nc*Nl);

%%% liblinear
%%% parameters of liblinear should be tuned based on your applications
h = hist(y,1:Nc);
h = max(h) ./ h;
str = [];
for i = 1:Nc
    str = [str, ' -w', num2str(i), ' ', num2str(h(i))];
end
L2_model = train(double(y),sparse(double(X)),['-s 1 -c 1e1 -B 1 -q', str]);


return;

