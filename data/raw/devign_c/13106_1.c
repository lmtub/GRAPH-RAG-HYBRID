static void ff_mpeg4_init_direct_mv(MpegEncContext *s){

    //FIXME table is stored in MpegEncContext for thread-safety,

    // but a static array would be faster

    static const int tab_size = sizeof(s->direct_scale_mv[0])/sizeof(int16_t);

    static const int tab_bias = (tab_size/2);

    int i;

    for(i=0; i<tab_size; i++){

        s->direct_scale_mv[0][i] = (i-tab_bias)*s->pb_time/s->pp_time;

        s->direct_scale_mv[1][i] = (i-tab_bias)*(s->pb_time-s->pp_time)/s->pp_time;

    }

}
