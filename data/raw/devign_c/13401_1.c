static int hds_write_header(AVFormatContext *s)

{

    HDSContext *c = s->priv_data;

    int ret = 0, i;

    AVOutputFormat *oformat;



    mkdir(s->filename, 0777);



    oformat = av_guess_format("flv", NULL, NULL);

    if (!oformat) {

        ret = AVERROR_MUXER_NOT_FOUND;

        goto fail;

    }



    c->streams = av_mallocz(sizeof(*c->streams) * s->nb_streams);

    if (!c->streams) {

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    for (i = 0; i < s->nb_streams; i++) {

        OutputStream *os = &c->streams[c->nb_streams];

        AVFormatContext *ctx;

        AVStream *st = s->streams[i];



        if (!st->codec->bit_rate) {

            av_log(s, AV_LOG_ERROR, "No bit rate set for stream %d\n", i);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO) {

            if (os->has_video) {

                c->nb_streams++;

                os++;

            }

            os->has_video = 1;

        } else if (st->codec->codec_type == AVMEDIA_TYPE_AUDIO) {

            if (os->has_audio) {

                c->nb_streams++;

                os++;

            }

            os->has_audio = 1;

        } else {

            av_log(s, AV_LOG_ERROR, "Unsupported stream type in stream %d\n", i);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        os->bitrate += s->streams[i]->codec->bit_rate;



        if (!os->ctx) {

            os->first_stream = i;

            ctx = avformat_alloc_context();

            if (!ctx) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

            os->ctx = ctx;

            ctx->oformat = oformat;

            ctx->interrupt_callback = s->interrupt_callback;



            ctx->pb = avio_alloc_context(os->iobuf, sizeof(os->iobuf),

                                         AVIO_FLAG_WRITE, os,

                                         NULL, hds_write, NULL);

            if (!ctx->pb) {

                ret = AVERROR(ENOMEM);

                goto fail;

            }

        } else {

            ctx = os->ctx;

        }

        s->streams[i]->id = c->nb_streams;



        if (!(st = avformat_new_stream(ctx, NULL))) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }

        avcodec_copy_context(st->codec, s->streams[i]->codec);

        st->sample_aspect_ratio = s->streams[i]->sample_aspect_ratio;

    }

    if (c->streams[c->nb_streams].ctx)

        c->nb_streams++;



    for (i = 0; i < c->nb_streams; i++) {

        OutputStream *os = &c->streams[i];

        int j;

        if ((ret = avformat_write_header(os->ctx, NULL)) < 0) {

             goto fail;

        }

        os->ctx_inited = 1;

        avio_flush(os->ctx->pb);

        for (j = 0; j < os->ctx->nb_streams; j++)

            s->streams[os->first_stream + j]->time_base = os->ctx->streams[j]->time_base;



        snprintf(os->temp_filename, sizeof(os->temp_filename),

                 "%s/stream%d_temp", s->filename, i);

        ret = init_file(s, os, 0);

        if (ret < 0)

            goto fail;



        if (!os->has_video && c->min_frag_duration <= 0) {

            av_log(s, AV_LOG_WARNING,

                   "No video stream in output stream %d and no min frag duration set\n", i);

            ret = AVERROR(EINVAL);

        }

        os->fragment_index = 1;

        write_abst(s, os, 0);

    }

    ret = write_manifest(s, 0);



fail:

    if (ret)

        hds_free(s);

    return ret;

}
