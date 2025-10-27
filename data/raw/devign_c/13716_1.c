static int wc3_read_close(AVFormatContext *s)

{

    Wc3DemuxContext *wc3 = s->priv_data;



    av_free(wc3->palettes);



    return 0;

}
