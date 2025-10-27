static void show_stream(WriterContext *w, AVFormatContext *fmt_ctx, int stream_idx)

{

    AVStream *stream = fmt_ctx->streams[stream_idx];

    AVCodecContext *dec_ctx;

    AVCodec *dec;

    char val_str[128];

    AVRational display_aspect_ratio;

    struct print_buf pbuf = {.s = NULL};



    print_section_header("stream");



    print_int("index", stream->index);



    if ((dec_ctx = stream->codec)) {

        if ((dec = dec_ctx->codec)) {

            print_str("codec_name",      dec->name);

            print_str("codec_long_name", dec->long_name);

        } else {

            print_str("codec_name",      "unknown");

        }



        print_str("codec_type", av_x_if_null(av_get_media_type_string(dec_ctx->codec_type), "unknown"));

        print_fmt("codec_time_base", "%d/%d", dec_ctx->time_base.num, dec_ctx->time_base.den);



        /* print AVI/FourCC tag */

        av_get_codec_tag_string(val_str, sizeof(val_str), dec_ctx->codec_tag);

        print_str("codec_tag_string",    val_str);

        print_fmt("codec_tag", "0x%04x", dec_ctx->codec_tag);



        switch (dec_ctx->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            print_int("width",        dec_ctx->width);

            print_int("height",       dec_ctx->height);

            print_int("has_b_frames", dec_ctx->has_b_frames);

            if (dec_ctx->sample_aspect_ratio.num) {

                print_fmt("sample_aspect_ratio", "%d:%d",

                          dec_ctx->sample_aspect_ratio.num,

                          dec_ctx->sample_aspect_ratio.den);

                av_reduce(&display_aspect_ratio.num, &display_aspect_ratio.den,

                          dec_ctx->width  * dec_ctx->sample_aspect_ratio.num,

                          dec_ctx->height * dec_ctx->sample_aspect_ratio.den,

                          1024*1024);

                print_fmt("display_aspect_ratio", "%d:%d",

                          display_aspect_ratio.num,

                          display_aspect_ratio.den);

            }

            print_str("pix_fmt", av_x_if_null(av_get_pix_fmt_name(dec_ctx->pix_fmt), "unknown"));

            print_int("level",   dec_ctx->level);

            break;



        case AVMEDIA_TYPE_AUDIO:

            print_str("sample_fmt",

                      av_x_if_null(av_get_sample_fmt_name(dec_ctx->sample_fmt), "unknown"));

            print_val("sample_rate",     dec_ctx->sample_rate, unit_hertz_str);

            print_int("channels",        dec_ctx->channels);

            print_int("bits_per_sample", av_get_bits_per_sample(dec_ctx->codec_id));

            break;

        }

    } else {

        print_str("codec_type", "unknown");

    }

    if (dec_ctx->codec && dec_ctx->codec->priv_class) {

        const AVOption *opt = NULL;

        while (opt = av_opt_next(dec_ctx->priv_data,opt)) {

            uint8_t *str;

            if (opt->flags) continue;

            if (av_opt_get(dec_ctx->priv_data, opt->name, 0, &str) >= 0) {

                print_str(opt->name, str);

                av_free(str);

            }

        }

    }



    if (fmt_ctx->iformat->flags & AVFMT_SHOW_IDS)

        print_fmt("id", "0x%x", stream->id);

    print_fmt("r_frame_rate",   "%d/%d", stream->r_frame_rate.num,   stream->r_frame_rate.den);

    print_fmt("avg_frame_rate", "%d/%d", stream->avg_frame_rate.num, stream->avg_frame_rate.den);

    print_fmt("time_base",      "%d/%d", stream->time_base.num,      stream->time_base.den);

    print_time("start_time",    stream->start_time, &stream->time_base);

    print_time("duration",      stream->duration,   &stream->time_base);

    if (stream->nb_frames)

        print_fmt("nb_frames", "%"PRId64, stream->nb_frames);



    show_tags(stream->metadata);



    print_section_footer("stream");

    av_free(pbuf.s);

    fflush(stdout);

}
