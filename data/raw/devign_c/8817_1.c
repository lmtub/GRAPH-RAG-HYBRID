static void add_input_streams(OptionsContext *o, AVFormatContext *ic)

{

    int i;



    for (i = 0; i < ic->nb_streams; i++) {

        AVStream *st = ic->streams[i];

        AVCodecContext *dec = st->codec;

        InputStream *ist = av_mallocz(sizeof(*ist));

        char *framerate = NULL, *hwaccel = NULL, *hwaccel_device = NULL;



        if (!ist)

            exit_program(1);



        GROW_ARRAY(input_streams, nb_input_streams);

        input_streams[nb_input_streams - 1] = ist;



        ist->st = st;

        ist->file_index = nb_input_files;

        ist->discard = 1;

        st->discard  = AVDISCARD_ALL;



        ist->ts_scale = 1.0;

        MATCH_PER_STREAM_OPT(ts_scale, dbl, ist->ts_scale, ic, st);



        ist->dec = choose_decoder(o, ic, st);

        ist->opts = filter_codec_opts(o->g->codec_opts, ist->st->codec->codec_id, ic, st, ist->dec);



        switch (dec->codec_type) {

        case AVMEDIA_TYPE_VIDEO:

            ist->resample_height  = dec->height;

            ist->resample_width   = dec->width;

            ist->resample_pix_fmt = dec->pix_fmt;



            MATCH_PER_STREAM_OPT(frame_rates, str, framerate, ic, st);

            if (framerate && av_parse_video_rate(&ist->framerate,

                                                 framerate) < 0) {

                av_log(NULL, AV_LOG_ERROR, "Error parsing framerate %s.\n",

                       framerate);

                exit_program(1);

            }



            MATCH_PER_STREAM_OPT(hwaccels, str, hwaccel, ic, st);

            if (hwaccel) {

                if (!strcmp(hwaccel, "none"))

                    ist->hwaccel_id = HWACCEL_NONE;

                else if (!strcmp(hwaccel, "auto"))

                    ist->hwaccel_id = HWACCEL_AUTO;

                else {

                    int i;

                    for (i = 0; hwaccels[i].name; i++) {

                        if (!strcmp(hwaccels[i].name, hwaccel)) {

                            ist->hwaccel_id = hwaccels[i].id;

                            break;

                        }

                    }



                    if (!ist->hwaccel_id) {

                        av_log(NULL, AV_LOG_FATAL, "Unrecognized hwaccel: %s.\n",

                               hwaccel);

                        av_log(NULL, AV_LOG_FATAL, "Supported hwaccels: ");

                        for (i = 0; hwaccels[i].name; i++)

                            av_log(NULL, AV_LOG_FATAL, "%s ", hwaccels[i].name);

                        av_log(NULL, AV_LOG_FATAL, "\n");

                        exit_program(1);

                    }

                }

            }



            MATCH_PER_STREAM_OPT(hwaccel_devices, str, hwaccel_device, ic, st);

            if (hwaccel_device) {

                ist->hwaccel_device = av_strdup(hwaccel_device);

                if (!ist->hwaccel_device)

                    exit_program(1);

            }




            break;

        case AVMEDIA_TYPE_AUDIO:

            guess_input_channel_layout(ist);



            ist->resample_sample_fmt     = dec->sample_fmt;

            ist->resample_sample_rate    = dec->sample_rate;

            ist->resample_channels       = dec->channels;

            ist->resample_channel_layout = dec->channel_layout;



            break;

        case AVMEDIA_TYPE_DATA:

        case AVMEDIA_TYPE_SUBTITLE:

        case AVMEDIA_TYPE_ATTACHMENT:

        case AVMEDIA_TYPE_UNKNOWN:

            break;

        default:

            abort();

        }

    }

}