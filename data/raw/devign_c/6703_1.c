static int ogg_write_trailer(AVFormatContext *s)

{

    int i;



    /* flush current page */

    for (i = 0; i < s->nb_streams; i++)

        ogg_buffer_page(s, s->streams[i]->priv_data);



    ogg_write_pages(s, 1);



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        OGGStreamContext *oggstream = st->priv_data;

        if (st->codec->codec_id == CODEC_ID_FLAC ||

            st->codec->codec_id == CODEC_ID_SPEEX) {

            av_free(oggstream->header[0]);


        }



        av_freep(&st->priv_data);

    }

    return 0;

}