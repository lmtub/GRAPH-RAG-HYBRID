static int write_trailer(AVFormatContext *s){

    NUTContext *nut= s->priv_data;

    AVIOContext *bc= s->pb;



    while(nut->header_count<3)

        write_headers(s, bc);

    avio_flush(bc);

    ff_nut_free_sp(nut);

    av_freep(&nut->stream);


    av_freep(&nut->time_base);



    return 0;

}