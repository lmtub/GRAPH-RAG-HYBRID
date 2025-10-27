PCA *ff_pca_init(int n){
    PCA *pca;
    if(n<=0)
    pca= av_mallocz(sizeof(*pca));
    pca->n= n;
    pca->z = av_malloc_array(n, sizeof(*pca->z));
    pca->count=0;
    pca->covariance= av_calloc(n*n, sizeof(double));
    pca->mean= av_calloc(n, sizeof(double));
    return pca;