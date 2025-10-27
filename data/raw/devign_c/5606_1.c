static int ogg_new_stream(AVFormatContext *s, uint32_t serial, int new_avstream)

{



    struct ogg *ogg = s->priv_data;

    int idx = ogg->nstreams++;

    AVStream *st;

    struct ogg_stream *os;



    ogg->streams = av_realloc (ogg->streams,

                               ogg->nstreams * sizeof (*ogg->streams));

    memset (ogg->streams + idx, 0, sizeof (*ogg->streams));

    os = ogg->streams + idx;

    os->serial = serial;

    os->bufsize = DECODER_BUFFER_SIZE;

    os->buf = av_malloc(os->bufsize);

    os->header = -1;



    if (new_avstream) {

        st = avformat_new_stream(s, NULL);

        if (!st)

            return AVERROR(ENOMEM);



        st->id = idx;

        avpriv_set_pts_info(st, 64, 1, 1000000);

    }



    return idx;

}
