% This is the source code published in
% J. Hou, H. Gao, X. Li. DSets-DBSCAN: A Parameter-Free Clustering
% Algorithm. IEEE Transactions on Image Processing, vol. 25, no. 7, 
% pp. 3182-3193. 2016.

function tip2016()

    %load the dataset
    fname='data\aggregation.txt';
    descr=dlmread(fname);
    dimen=size(descr,2);
    label_t=descr(:,dimen);
    descr=descr(:,1:dimen-1);
    
    %build the similarity matrix
    sigma=1;
    dist=pdist(descr,'euclidean');
    dima=squareform(dist);
    
    d_mean=mean2(dima);
    sima=exp(-dima/(d_mean*sigma));
    clear dima descr;
    
    %transformation by histogram equalization
    sima=sima_equalize(sima,100);
    
    %clustering
    paras.toll=1e-4;
    paras.th=0;
    label_c=dsetpp_extend_dbscan2(sima,paras);
   
    %results
    score_fmeasure=label2fmeasure(label_c,label_t)

end


function label=dsetpp_extend_dbscan2(sima,paras_dset)

    toll=paras_dset.toll;
    th_weight=paras_dset.th;
    
    %data
    dima=-log(sima);
    dsima=size(sima,1);
    label=zeros(1,dsima);
    
    min_size=3;
    th_size=min_size+1;             %the minimum size of a cluster
            
    %dset initialization
    for i=1:dsima
        sima(i,i)=0;
    end
    x=zeros(1,dsima);
    x(:)=1/dsima;
    
    %start clustering
    num_dsets=0;

    while 1>0
        if sum(label==0)<5
            break;
        end

        %dset extraction
        x=indyn(sima,x,toll);
        idx_dset=find(x>th_weight);
        
        if length(idx_dset)<th_size
            break;
        end
        
        %dbscan based extension
        num_dsets=num_dsets+1;
        label(idx_dset)=num_dsets;
        
        label=extend_density_dima(dima,idx_dset,label,num_dsets,min_size);
        
        %new sima for extracting the next dset
        idx=label>0;
        sima(idx,:)=0;
        sima(:,idx)=0;
        
        %new x for extracting the next dset
        idx_ndset=find(label==0);
        num_ndset=length(idx_ndset);
        x(:)=0;
        x(idx_ndset)=1/num_ndset;
    end

end

function sima_n=sima_equalize(sima_o,dhist)

    dsima=size(sima_o,1);

    if dhist==0
        simv_o=reshape(sima_o,1,dsima*dsima);
        [~,idx]=sort(simv_o,'ascend');
        sima_n=reshape(idx,dsima,dsima);
        sima_n=sima_n/(dsima*dsima);
    else
        hist=zeros(1,dhist);
        dsima=size(sima_o,1);

        %the max and min similarity
        smax=max(max(sima_o));
        smin=min(min(sima_o));

        %build the hist
        bin=(smax-smin)/dhist;
        flag=zeros(dsima,dsima);
    
        for i=1:dsima
            for j=1:dsima
                if i~=j
                    no=min(floor((sima_o(i,j)-smin)/bin)+1,dhist);
                    hist(no)=hist(no)+1;
                    flag(i,j)=no;
                end
            end
        end
        
        %hist normalization
        count=zeros(1,dhist);
    
        for i=1:dhist
            if i==1
                count(i)=hist(1);
            else
                count(i)=count(i-1)+hist(i);
            end
        end
        sim=count/sum(hist);
    
        %sima normalization
        sima_n=zeros(dsima,dsima);
    
        range=zeros(dhist,2);
        for i=1:dhist
            idx=flag==i;

            if sum(sum(idx))>0
                range(i,1)=min(min(sima_o(idx)));
                range(i,2)=max(max(sima_o(idx)));
            end
        end
        
        for i=1:dhist
            if i==1
                vmin=0;
            else
                vmin=sim(i-1);
            end
            vmax=sim(i);
        
            idx=find(flag==i)';
            for j=idx
                if range(i,1)==range(i,2)
                    sima_n(j)=(vmax+vmin)/2;
                else
                    sima_n(j)=vmin+(vmax-vmin)/(range(i,2)-range(i,1))*(sima_o(j)-range(i,1));
                end
            end
        end
        
        for i=1:dsima
            sima_n(i,i)=1;
        end
        
        clear range flag sima_o;
    end

end


function score=label2fmeasure(label_c,label_t)

    vlabel_c=unique(label_c);
    nlabel_c=length(vlabel_c);
    
    vlabel_t=unique(label_t);
    nlabel_t=length(vlabel_t);
    
    ndata=length(label_t);

    score=0;
    for i=1:nlabel_t
        lt=vlabel_t(i);
        nlt=length(find(label_t==lt));
        
        sf=zeros(1,nlabel_c);
        
        for j=1:nlabel_c
            lc=vlabel_c(j);
            nlc=length(find(label_c==lc));
            
            num=0;
            for k=1:ndata
                if label_t(k)==lt && label_c(k)==lc
                    num=num+1;
                end
            end
            
            sp=num/nlc;
            sr=num/nlt;
            sf(j)=2*sp*sr/(sp+sr);
        end
        
        score=score+nlt*max(sf);
        clear sf;
    end

    score=score/ndata;

end

function x=indyn(sima,x,toll)

    dsima=size(sima,1);
    if (~exist('x','var'))
        x=zeros(dsima,1);
        maxv=max(sima);
        for i=1:dsima
            if maxv(i)>0
                x(i)=1;
                break;
            end
        end
    end
    
    if (~exist('toll','var'))
        toll=0.005;
    end
    
    for i=1:dsima
        sima(i,i)=0;
    end
    
    x=reshape(x,dsima,1);

    %start operation
    g = sima*x;
    AT = sima';
    h = AT*x;
    niter=0;
    while 1,
        r = g - (x'*g);
        if norm(min(x,-r))<toll
            break;
        end
        i = selectPureStrategy(x,r);
        den = sima(i,i) - h(i) - r(i); %In case of asymmetric affinities
        do_remove=0;
        if r(i)>=0
            mu = 1;
            if den<0
                mu = min(mu, -r(i)/den);
                if mu<0 
                    mu=0; 
                end
            end
        else
            do_remove=1;
            mu = x(i)/(x(i)-1);
            if den<0
                [mu do_remove] = max([mu -r(i)/den]);
                do_remove=do_remove==1;
            end
        end
        tmp = -x;
        tmp(i) = tmp(i)+1;
        x = mu*tmp + x;
        if(do_remove) 
           x(i)=0; 
        end;
        x=abs(x)/sum(abs(x));
        
        g = mu*(sima(:,i)-g) + g;
        h = mu*(AT(:,i)-h) + h; %In case of asymmetric affinities
        niter=niter+1;
    end
    
    x=x';
end

function [i] = selectPureStrategy(x,r)
    index=1:length(x);
    mask = x>0;
    masked_index = index(mask);
    [~, i] = max(r);
    [~, j] = min(r(x>0));
    j = masked_index(j);
    if r(i)<-r(j)
        i = j;
    end
    return;
end

function label=extend_density_dima(dima,idx_dset,label,num_dsets,min_size)

    touched=zeros(1,length(label));
    sub_dima=dima(idx_dset,idx_dset);
    
    sub_dima1=sort(sub_dima,2,'ascend');
    sim_min=sub_dima1(:,min_size+1);
    th_max=max(sim_min);    %the max disimilarity of one member with min_size nearest neighbors
    
    %search neighborhood
    for ii=1:length(idx_dset)    
        idx_core=idx_dset(ii);
        touched(idx_core)=1;
    
        dist=dima(idx_core,:);
        ind=find(dist<=th_max);
    
        if length(ind)>=min_size+1
            label(idx_core)=num_dsets;
        
            while ~isempty(ind)
                idx=ind(1);
                ind(1)=[];
            
                if touched(idx)==0
                    touched(idx)=1;
                
                    dist=dima(idx,:);
                    i1=find(dist<=th_max);
                
                    if length(i1)>=min_size+1
                        ind=[ind i1];
                        ind=unique(ind);
                    elseif length(i1)>=1
                        label(i1)=num_dsets;
                    end
                end
                
                if label(idx)==0
                    label(idx)=num_dsets;
                end
            end
        end
    end

end