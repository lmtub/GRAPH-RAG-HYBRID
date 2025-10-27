static void add_input_streams(OptionsContext *o, AVFormatContext *ic)

{

    int i;

    char *next, *codec_tag = NULL;



    for (i = 0; i < ic->nb_streams; i++) {

        AVStream *st = ic->streams[i];

        AVCodecContext *dec = st->codec;

        InputStream *ist = av_mallocz(sizeof(*ist));

        char *framerate = NULL;



        if (!ist)

            exit(1);



        GROW_ARRAY(input_streams, nb_input_streams);

        input_streams[nb_input_streams - 1] = ist;



        ist->st = st;

        ist->file_index = nb_input_files;

        ist->discard = 1;

        st->discard  = AVDISCARD_ALL;



        ist->ts_scale = 1.0;

        MATCH_PER_STREAM_OPT(ts_scale, dbl, ist->ts_scale, ic, st);



        MATCH_PER_STREAM_OPT(codec_tags, str, codec_tag, ic, st);

        if (codec_tag) {

            uint32_t tag = strtol(codec_tag, &next, 0);

            if (*next)

                tag = AV_RL32(codec_tag);

            st->codec->codec_tag = tag;

        }



        ist->dec = choose_decoder(o, ic, st);

        ist->opts = filter_codec_opts(o->g->codec_opts, ist->st->codec->codec_id, ic, st, ist->dec);



        ist->reinit_filters = -1;

        MATCH_PER_STREAM_OPT(reinit_filters, i, ist->reinit_filters, ic, st);



        ist->filter_in_rescale_delta_last = AV_NOPTS_VALUE;



        switch (dec->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            if(!ist->dec)

                ist->dec = avcodec_find_decoder(dec->codec_id);

            if (dec->lowres) {

                dec->flags |= CODEC_FLAG_EMU_EDGE;

            }



            ist->resample_height  = dec->height;

            ist->resample_width   = dec->width;

            ist->resample_pix_fmt = dec->pix_fmt;



            MATCH_PER_STREAM_OPT(frame_rates, str, framerate, ic, st);

            if (framerate && av_parse_video_rate(&ist->framerate,

                                                 framerate) < 0) {

                av_log(NULL, AV_LOG_ERROR, "Error parsing framerate %s.\n",

                       framerate);

                exit(1);

            }



            ist->top_field_first = -1;

            MATCH_PER_STREAM_OPT(top_field_first, i, ist->top_field_first, ic, st);



            break;

        case AVMEDIA_TYPE_AUDIO:

            ist->guess_layout_max = INT_MAX;

            MATCH_PER_STREAM_OPT(guess_layout_max, i, ist->guess_layout_max, ic, st);

            guess_input_channel_layout(ist);



            ist->resample_sample_fmt     = dec->sample_fmt;

            ist->resample_sample_rate    = dec->sample_rate;

            ist->resample_channels       = dec->channels;

            ist->resample_channel_layout = dec->channel_layout;



            break;

        case AVMEDIA_TYPE_DATA:

        case AVMEDIA_TYPE_SUBTITLE:

            if(!ist->dec)

                ist->dec = avcodec_find_decoder(dec->codec_id);

            MATCH_PER_STREAM_OPT(fix_sub_duration, i, ist->fix_sub_duration, ic, st);

            break;

        case AVMEDIA_TYPE_ATTACHMENT:

        case AVMEDIA_TYPE_UNKNOWN:

            break;

        default:

            abort();

        }

    }

}
