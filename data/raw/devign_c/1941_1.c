static int common_end(AVCodecContext *avctx){

    FFV1Context *s = avctx->priv_data;

    int i;



    for(i=0; i<s->plane_count; i++){

        PlaneContext *p= &s->plane[i];



        av_freep(&p->state);


    }



    return 0;

}