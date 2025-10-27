static int parse_video_var(AVFormatContext *avctx, AVStream *st, const char *name, int size)

{

    AVIOContext *pb = avctx->pb;

    if (!strcmp(name, "__DIR_COUNT")) {

        st->nb_frames = st->duration = var_read_int(pb, size);

    } else if (!strcmp(name, "COMPRESSION")) {

        char * str = var_read_string(pb, size);

        if (!strcmp(str, "1")) {

            st->codec->codec_id = AV_CODEC_ID_MVC1;

        } else if (!strcmp(str, "2")) {

            st->codec->pix_fmt  = AV_PIX_FMT_ABGR;

            st->codec->codec_id = AV_CODEC_ID_RAWVIDEO;

        } else if (!strcmp(str, "3")) {

            st->codec->codec_id = AV_CODEC_ID_SGIRLE;

        } else if (!strcmp(str, "10")) {

            st->codec->codec_id = AV_CODEC_ID_MJPEG;

        } else if (!strcmp(str, "MVC2")) {

            st->codec->codec_id = AV_CODEC_ID_MVC2;

        } else {

            av_log_ask_for_sample(avctx, "unknown video compression %s\n", str);

        }

        av_free(str);

    } else if (!strcmp(name, "FPS")) {

        st->time_base = av_inv_q(var_read_float(pb, size));

    } else if (!strcmp(name, "HEIGHT")) {

        st->codec->height = var_read_int(pb, size);

    } else if (!strcmp(name, "PIXEL_ASPECT")) {

        st->sample_aspect_ratio = var_read_float(pb, size);

        av_reduce(&st->sample_aspect_ratio.num, &st->sample_aspect_ratio.den,

                  st->sample_aspect_ratio.num, st->sample_aspect_ratio.den, INT_MAX);

    } else if (!strcmp(name, "WIDTH")) {

        st->codec->width = var_read_int(pb, size);

    } else if (!strcmp(name, "ORIENTATION")) {

        if (var_read_int(pb, size) == 1101) {

            st->codec->extradata       = av_strdup("BottomUp");

            st->codec->extradata_size  = 9;

        }

    } else if (!strcmp(name, "Q_SPATIAL") || !strcmp(name, "Q_TEMPORAL")) {

        var_read_metadata(avctx, name, size);

    } else if (!strcmp(name, "INTERLACING") || !strcmp(name, "PACKING")) {

        avio_skip(pb, size); // ignore

    } else

        return -1;

    return 0;

}
