static int webm_dash_manifest_write_trailer(AVFormatContext *s)

{

    WebMDashMuxContext *w = s->priv_data;

    int i;

    for (i = 0; i < w->nb_as; i++) {

        av_freep(&w->as[i].streams);

    }

    av_freep(&w->as);

    return 0;

}
