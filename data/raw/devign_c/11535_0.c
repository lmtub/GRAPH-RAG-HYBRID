static void show_stream(WriterContext *w, AVFormatContext *fmt_ctx, int stream_idx, int in_program)

{

    AVStream *stream = fmt_ctx->streams[stream_idx];

    AVCodecContext *dec_ctx;

    const AVCodec *dec;

    char val_str[128];

    const char *s;

    AVRational sar, dar;

    AVBPrint pbuf;



    av_bprint_init(&pbuf, 1, AV_BPRINT_SIZE_UNLIMITED);



    writer_print_section_header(w, in_program ? SECTION_ID_PROGRAM_STREAM : SECTION_ID_STREAM);



    print_int("index", stream->index);



    if ((dec_ctx = stream->codec)) {

        const char *profile = NULL;

        dec = dec_ctx->codec;

        if (dec) {

            print_str("codec_name", dec->name);

            if (!do_bitexact) {

                if (dec->long_name) print_str    ("codec_long_name", dec->long_name);

                else                print_str_opt("codec_long_name", "unknown");

            }

        } else {

            print_str_opt("codec_name", "unknown");

            if (!do_bitexact) {

                print_str_opt("codec_long_name", "unknown");

            }

        }



        if (dec && (profile = av_get_profile_name(dec, dec_ctx->profile)))

            print_str("profile", profile);

        else

            print_str_opt("profile", "unknown");



        s = av_get_media_type_string(dec_ctx->codec_type);

        if (s) print_str    ("codec_type", s);

        else   print_str_opt("codec_type", "unknown");

        print_q("codec_time_base", dec_ctx->time_base, '/');



        /* print AVI/FourCC tag */

        av_get_codec_tag_string(val_str, sizeof(val_str), dec_ctx->codec_tag);

        print_str("codec_tag_string",    val_str);

        print_fmt("codec_tag", "0x%04x", dec_ctx->codec_tag);



        switch (dec_ctx->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            print_int("width",        dec_ctx->width);

            print_int("height",       dec_ctx->height);

            print_int("has_b_frames", dec_ctx->has_b_frames);

            sar = av_guess_sample_aspect_ratio(fmt_ctx, stream, NULL);

            if (sar.den) {

                print_q("sample_aspect_ratio", sar, ':');

                av_reduce(&dar.num, &dar.den,

                          dec_ctx->width  * sar.num,

                          dec_ctx->height * sar.den,

                          1024*1024);

                print_q("display_aspect_ratio", dar, ':');

            } else {

                print_str_opt("sample_aspect_ratio", "N/A");

                print_str_opt("display_aspect_ratio", "N/A");

            }

            s = av_get_pix_fmt_name(dec_ctx->pix_fmt);

            if (s) print_str    ("pix_fmt", s);

            else   print_str_opt("pix_fmt", "unknown");

            print_int("level",   dec_ctx->level);

            if (dec_ctx->timecode_frame_start >= 0) {

                char tcbuf[AV_TIMECODE_STR_SIZE];

                av_timecode_make_mpeg_tc_string(tcbuf, dec_ctx->timecode_frame_start);

                print_str("timecode", tcbuf);

            } else {

                print_str_opt("timecode", "N/A");

            }

            break;



        case AVMEDIA_TYPE_AUDIO:

            s = av_get_sample_fmt_name(dec_ctx->sample_fmt);

            if (s) print_str    ("sample_fmt", s);

            else   print_str_opt("sample_fmt", "unknown");

            print_val("sample_rate",     dec_ctx->sample_rate, unit_hertz_str);

            print_int("channels",        dec_ctx->channels);



            if (dec_ctx->channel_layout) {

                av_bprint_clear(&pbuf);

                av_bprint_channel_layout(&pbuf, dec_ctx->channels, dec_ctx->channel_layout);

                print_str    ("channel_layout", pbuf.str);

            } else {

                print_str_opt("channel_layout", "unknown");

            }



            print_int("bits_per_sample", av_get_bits_per_sample(dec_ctx->codec_id));

            break;



        case AVMEDIA_TYPE_SUBTITLE:

            if (dec_ctx->width)

                print_int("width",       dec_ctx->width);

            else

                print_str_opt("width",   "N/A");

            if (dec_ctx->height)

                print_int("height",      dec_ctx->height);

            else

                print_str_opt("height",  "N/A");

            break;

        }

    } else {

        print_str_opt("codec_type", "unknown");

    }

    if (dec_ctx->codec && dec_ctx->codec->priv_class && show_private_data) {

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



    if (fmt_ctx->iformat->flags & AVFMT_SHOW_IDS) print_fmt    ("id", "0x%x", stream->id);

    else                                          print_str_opt("id", "N/A");

    print_q("r_frame_rate",   stream->r_frame_rate,   '/');

    print_q("avg_frame_rate", stream->avg_frame_rate, '/');

    print_q("time_base",      stream->time_base,      '/');

    print_ts  ("start_pts",   stream->start_time);

    print_time("start_time",  stream->start_time, &stream->time_base);

    print_ts  ("duration_ts", stream->duration);

    print_time("duration",    stream->duration, &stream->time_base);

    if (dec_ctx->bit_rate > 0) print_val    ("bit_rate", dec_ctx->bit_rate, unit_bit_per_second_str);

    else                       print_str_opt("bit_rate", "N/A");

    if (stream->nb_frames) print_fmt    ("nb_frames", "%"PRId64, stream->nb_frames);

    else                   print_str_opt("nb_frames", "N/A");

    if (nb_streams_frames[stream_idx])  print_fmt    ("nb_read_frames", "%"PRIu64, nb_streams_frames[stream_idx]);

    else                                print_str_opt("nb_read_frames", "N/A");

    if (nb_streams_packets[stream_idx]) print_fmt    ("nb_read_packets", "%"PRIu64, nb_streams_packets[stream_idx]);

    else                                print_str_opt("nb_read_packets", "N/A");

    if (do_show_data)

        writer_print_data(w, "extradata", dec_ctx->extradata,

                                          dec_ctx->extradata_size);



    /* Print disposition information */

#define PRINT_DISPOSITION(flagname, name) do {                                \

        print_int(name, !!(stream->disposition & AV_DISPOSITION_##flagname)); \

    } while (0)



    if (do_show_stream_disposition) {

    writer_print_section_header(w, in_program ? SECTION_ID_PROGRAM_STREAM_DISPOSITION : SECTION_ID_STREAM_DISPOSITION);

    PRINT_DISPOSITION(DEFAULT,          "default");

    PRINT_DISPOSITION(DUB,              "dub");

    PRINT_DISPOSITION(ORIGINAL,         "original");

    PRINT_DISPOSITION(COMMENT,          "comment");

    PRINT_DISPOSITION(LYRICS,           "lyrics");

    PRINT_DISPOSITION(KARAOKE,          "karaoke");

    PRINT_DISPOSITION(FORCED,           "forced");

    PRINT_DISPOSITION(HEARING_IMPAIRED, "hearing_impaired");

    PRINT_DISPOSITION(VISUAL_IMPAIRED,  "visual_impaired");

    PRINT_DISPOSITION(CLEAN_EFFECTS,    "clean_effects");

    PRINT_DISPOSITION(ATTACHED_PIC,     "attached_pic");

    writer_print_section_footer(w);

    }



    show_tags(w, stream->metadata, in_program ? SECTION_ID_PROGRAM_STREAM_TAGS : SECTION_ID_STREAM_TAGS);



    writer_print_section_footer(w);

    av_bprint_finalize(&pbuf, NULL);

    fflush(stdout);

}
