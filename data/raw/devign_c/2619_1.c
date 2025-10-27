static int ogg_read_header(AVFormatContext *s)

{

    struct ogg *ogg = s->priv_data;

    int ret, i;



    ogg->curidx = -1;



    //linear headers seek from start

    do {

        ret = ogg_packet(s, NULL, NULL, NULL, NULL);

        if (ret < 0) {

            ogg_read_close(s);

            return ret;

        }

    } while (!ogg->headers);

    av_log(s, AV_LOG_TRACE, "found headers\n");



    for (i = 0; i < ogg->nstreams; i++) {

        struct ogg_stream *os = ogg->streams + i;



        if (ogg->streams[i].header < 0) {

            av_log(s, AV_LOG_ERROR, "Header parsing failed for stream %d\n", i);

            ogg->streams[i].codec = NULL;


        } else if (os->codec && os->nb_header < os->codec->nb_header) {

            av_log(s, AV_LOG_WARNING,

                   "Headers mismatch for stream %d: "

                   "expected %d received %d.\n",

                   i, os->codec->nb_header, os->nb_header);

            if (s->error_recognition & AV_EF_EXPLODE)

                return AVERROR_INVALIDDATA;

        }

        if (os->start_granule != OGG_NOGRANULE_VALUE)

            os->lastpts = s->streams[i]->start_time =

                ogg_gptopts(s, i, os->start_granule, NULL);

    }



    //linear granulepos seek from end

    ret = ogg_get_length(s);

    if (ret < 0) {

        ogg_read_close(s);

        return ret;

    }



    return 0;

}