AVBitStreamFilterContext *av_bitstream_filter_init(const char *name){

    AVBitStreamFilter *bsf= first_bitstream_filter;



    while(bsf){

        if(!strcmp(name, bsf->name)){

            AVBitStreamFilterContext *bsfc= av_mallocz(sizeof(AVBitStreamFilterContext));

            bsfc->filter= bsf;

            bsfc->priv_data= av_mallocz(bsf->priv_data_size);

            return bsfc;

        }

        bsf= bsf->next;

    }

    return NULL;

}
