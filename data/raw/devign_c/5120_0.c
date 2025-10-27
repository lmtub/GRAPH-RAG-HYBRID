static int transcode_init(void)

{

    int ret = 0, i, j, k;

    AVFormatContext *oc;

    OutputStream *ost;

    InputStream *ist;

    char error[1024] = {0};

    int want_sdp = 1;



    for (i = 0; i < nb_filtergraphs; i++) {

        FilterGraph *fg = filtergraphs[i];

        for (j = 0; j < fg->nb_outputs; j++) {

            OutputFilter *ofilter = fg->outputs[j];

            if (!ofilter->ost || ofilter->ost->source_index >= 0)

                continue;

            if (fg->nb_inputs != 1)

                continue;

            for (k = nb_input_streams-1; k >= 0 ; k--)

                if (fg->inputs[0]->ist == input_streams[k])

                    break;

            ofilter->ost->source_index = k;

        }

    }



    /* init framerate emulation */

    for (i = 0; i < nb_input_files; i++) {

        InputFile *ifile = input_files[i];

        if (ifile->rate_emu)

            for (j = 0; j < ifile->nb_streams; j++)

                input_streams[j + ifile->ist_index]->start = av_gettime_relative();

    }



    /* output stream init */

    for (i = 0; i < nb_output_files; i++) {

        oc = output_files[i]->ctx;

        if (!oc->nb_streams && !(oc->oformat->flags & AVFMT_NOSTREAMS)) {

            av_dump_format(oc, i, oc->filename, 1);

            av_log(NULL, AV_LOG_ERROR, "Output file #%d does not contain any stream\n", i);

            return AVERROR(EINVAL);

        }

    }



    /* init complex filtergraphs */

    for (i = 0; i < nb_filtergraphs; i++)

        if ((ret = avfilter_graph_config(filtergraphs[i]->graph, NULL)) < 0)

            return ret;



    /* for each output stream, we compute the right encoding parameters */

    for (i = 0; i < nb_output_streams; i++) {

        AVCodecContext *enc_ctx;

        AVCodecContext *dec_ctx = NULL;

        ost = output_streams[i];

        oc  = output_files[ost->file_index]->ctx;

        ist = get_input_stream(ost);



        if (ost->attachment_filename)

            continue;



        enc_ctx = ost->stream_copy ? ost->st->codec : ost->enc_ctx;



        if (ist) {

            dec_ctx = ist->dec_ctx;



            ost->st->disposition          = ist->st->disposition;

            enc_ctx->bits_per_raw_sample    = dec_ctx->bits_per_raw_sample;

            enc_ctx->chroma_sample_location = dec_ctx->chroma_sample_location;

        } else {

            for (j=0; j<oc->nb_streams; j++) {

                AVStream *st = oc->streams[j];

                if (st != ost->st && st->codec->codec_type == enc_ctx->codec_type)

                    break;

            }

            if (j == oc->nb_streams)

                if (enc_ctx->codec_type == AVMEDIA_TYPE_AUDIO || enc_ctx->codec_type == AVMEDIA_TYPE_VIDEO)

                    ost->st->disposition = AV_DISPOSITION_DEFAULT;

        }



        if (ost->stream_copy) {

            AVRational sar;

            uint64_t extra_size;



            av_assert0(ist && !ost->filter);



            extra_size = (uint64_t)dec_ctx->extradata_size + FF_INPUT_BUFFER_PADDING_SIZE;



            if (extra_size > INT_MAX) {

                return AVERROR(EINVAL);

            }



            /* if stream_copy is selected, no need to decode or encode */

            enc_ctx->codec_id   = dec_ctx->codec_id;

            enc_ctx->codec_type = dec_ctx->codec_type;



            if (!enc_ctx->codec_tag) {

                unsigned int codec_tag;

                if (!oc->oformat->codec_tag ||

                     av_codec_get_id (oc->oformat->codec_tag, dec_ctx->codec_tag) == enc_ctx->codec_id ||

                     !av_codec_get_tag2(oc->oformat->codec_tag, dec_ctx->codec_id, &codec_tag))

                    enc_ctx->codec_tag = dec_ctx->codec_tag;

            }



            enc_ctx->bit_rate       = dec_ctx->bit_rate;

            enc_ctx->rc_max_rate    = dec_ctx->rc_max_rate;

            enc_ctx->rc_buffer_size = dec_ctx->rc_buffer_size;

            enc_ctx->field_order    = dec_ctx->field_order;

            if (dec_ctx->extradata_size) {

                enc_ctx->extradata      = av_mallocz(extra_size);

                if (!enc_ctx->extradata) {

                    return AVERROR(ENOMEM);

                }

                memcpy(enc_ctx->extradata, dec_ctx->extradata, dec_ctx->extradata_size);

            }

            enc_ctx->extradata_size= dec_ctx->extradata_size;

            enc_ctx->bits_per_coded_sample  = dec_ctx->bits_per_coded_sample;



            enc_ctx->time_base = ist->st->time_base;

            /*

             * Avi is a special case here because it supports variable fps but

             * having the fps and timebase differe significantly adds quite some

             * overhead

             */

            if(!strcmp(oc->oformat->name, "avi")) {

                if ( copy_tb<0 && av_q2d(ist->st->r_frame_rate) >= av_q2d(ist->st->avg_frame_rate)

                               && 0.5/av_q2d(ist->st->r_frame_rate) > av_q2d(ist->st->time_base)

                               && 0.5/av_q2d(ist->st->r_frame_rate) > av_q2d(dec_ctx->time_base)

                               && av_q2d(ist->st->time_base) < 1.0/500 && av_q2d(dec_ctx->time_base) < 1.0/500

                     || copy_tb==2){

                    enc_ctx->time_base.num = ist->st->r_frame_rate.den;

                    enc_ctx->time_base.den = 2*ist->st->r_frame_rate.num;

                    enc_ctx->ticks_per_frame = 2;

                } else if (   copy_tb<0 && av_q2d(dec_ctx->time_base)*dec_ctx->ticks_per_frame > 2*av_q2d(ist->st->time_base)

                                 && av_q2d(ist->st->time_base) < 1.0/500

                    || copy_tb==0){

                    enc_ctx->time_base = dec_ctx->time_base;

                    enc_ctx->time_base.num *= dec_ctx->ticks_per_frame;

                    enc_ctx->time_base.den *= 2;

                    enc_ctx->ticks_per_frame = 2;

                }

            } else if(!(oc->oformat->flags & AVFMT_VARIABLE_FPS)

                      && strcmp(oc->oformat->name, "mov") && strcmp(oc->oformat->name, "mp4") && strcmp(oc->oformat->name, "3gp")

                      && strcmp(oc->oformat->name, "3g2") && strcmp(oc->oformat->name, "psp") && strcmp(oc->oformat->name, "ipod")

                      && strcmp(oc->oformat->name, "f4v")

            ) {

                if(   copy_tb<0 && dec_ctx->time_base.den

                                && av_q2d(dec_ctx->time_base)*dec_ctx->ticks_per_frame > av_q2d(ist->st->time_base)

                                && av_q2d(ist->st->time_base) < 1.0/500

                   || copy_tb==0){

                    enc_ctx->time_base = dec_ctx->time_base;

                    enc_ctx->time_base.num *= dec_ctx->ticks_per_frame;

                }

            }

            if (   enc_ctx->codec_tag == AV_RL32("tmcd")

                && dec_ctx->time_base.num < dec_ctx->time_base.den

                && dec_ctx->time_base.num > 0

                && 121LL*dec_ctx->time_base.num > dec_ctx->time_base.den) {

                enc_ctx->time_base = dec_ctx->time_base;

            }



            if (ist && !ost->frame_rate.num)

                ost->frame_rate = ist->framerate;

            if(ost->frame_rate.num)

                enc_ctx->time_base = av_inv_q(ost->frame_rate);



            av_reduce(&enc_ctx->time_base.num, &enc_ctx->time_base.den,

                        enc_ctx->time_base.num, enc_ctx->time_base.den, INT_MAX);



            if (ist->st->nb_side_data) {

                ost->st->side_data = av_realloc_array(NULL, ist->st->nb_side_data,

                                                      sizeof(*ist->st->side_data));

                if (!ost->st->side_data)

                    return AVERROR(ENOMEM);



                for (j = 0; j < ist->st->nb_side_data; j++) {

                    const AVPacketSideData *sd_src = &ist->st->side_data[j];

                    AVPacketSideData *sd_dst = &ost->st->side_data[j];



                    sd_dst->data = av_malloc(sd_src->size);

                    if (!sd_dst->data)

                        return AVERROR(ENOMEM);

                    memcpy(sd_dst->data, sd_src->data, sd_src->size);

                    sd_dst->size = sd_src->size;

                    sd_dst->type = sd_src->type;

                    ost->st->nb_side_data++;

                }

            }



            ost->parser = av_parser_init(enc_ctx->codec_id);



            switch (enc_ctx->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                if (audio_volume != 256) {

                    av_log(NULL, AV_LOG_FATAL, "-acodec copy and -vol are incompatible (frames are not decoded)\n");

                    exit_program(1);

                }

                enc_ctx->channel_layout     = dec_ctx->channel_layout;

                enc_ctx->sample_rate        = dec_ctx->sample_rate;

                enc_ctx->channels           = dec_ctx->channels;

                enc_ctx->frame_size         = dec_ctx->frame_size;

                enc_ctx->audio_service_type = dec_ctx->audio_service_type;

                enc_ctx->block_align        = dec_ctx->block_align;

                enc_ctx->initial_padding    = dec_ctx->delay;

#if FF_API_AUDIOENC_DELAY

                enc_ctx->delay              = dec_ctx->delay;

#endif

                if((enc_ctx->block_align == 1 || enc_ctx->block_align == 1152 || enc_ctx->block_align == 576) && enc_ctx->codec_id == AV_CODEC_ID_MP3)

                    enc_ctx->block_align= 0;

                if(enc_ctx->codec_id == AV_CODEC_ID_AC3)

                    enc_ctx->block_align= 0;

                break;

            case AVMEDIA_TYPE_VIDEO:

                enc_ctx->pix_fmt            = dec_ctx->pix_fmt;

                enc_ctx->width              = dec_ctx->width;

                enc_ctx->height             = dec_ctx->height;

                enc_ctx->has_b_frames       = dec_ctx->has_b_frames;

                if (ost->frame_aspect_ratio.num) { // overridden by the -aspect cli option

                    sar =

                        av_mul_q(ost->frame_aspect_ratio,

                                 (AVRational){ enc_ctx->height, enc_ctx->width });

                    av_log(NULL, AV_LOG_WARNING, "Overriding aspect ratio "

                           "with stream copy may produce invalid files\n");

                }

                else if (ist->st->sample_aspect_ratio.num)

                    sar = ist->st->sample_aspect_ratio;

                else

                    sar = dec_ctx->sample_aspect_ratio;

                ost->st->sample_aspect_ratio = enc_ctx->sample_aspect_ratio = sar;

                ost->st->avg_frame_rate = ist->st->avg_frame_rate;

                ost->st->r_frame_rate = ist->st->r_frame_rate;

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                enc_ctx->width  = dec_ctx->width;

                enc_ctx->height = dec_ctx->height;

                break;

            case AVMEDIA_TYPE_DATA:

            case AVMEDIA_TYPE_ATTACHMENT:

                break;

            default:

                abort();

            }

        } else {

            if (!ost->enc)

                ost->enc = avcodec_find_encoder(enc_ctx->codec_id);

            if (!ost->enc) {

                /* should only happen when a default codec is not present. */

                snprintf(error, sizeof(error), "Encoder (codec %s) not found for output stream #%d:%d",

                         avcodec_get_name(ost->st->codec->codec_id), ost->file_index, ost->index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }



            if (ist)

                ist->decoding_needed |= DECODING_FOR_OST;

            ost->encoding_needed = 1;



            set_encoder_id(output_files[ost->file_index], ost);



            if (!ost->filter &&

                (enc_ctx->codec_type == AVMEDIA_TYPE_VIDEO ||

                 enc_ctx->codec_type == AVMEDIA_TYPE_AUDIO)) {

                    FilterGraph *fg;

                    fg = init_simple_filtergraph(ist, ost);

                    if (configure_filtergraph(fg)) {

                        av_log(NULL, AV_LOG_FATAL, "Error opening filters!\n");

                        exit_program(1);

                    }

            }



            if (enc_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {

                if (ost->filter && !ost->frame_rate.num)

                    ost->frame_rate = av_buffersink_get_frame_rate(ost->filter->filter);

                if (ist && !ost->frame_rate.num)

                    ost->frame_rate = ist->framerate;

                if (ist && !ost->frame_rate.num)

                    ost->frame_rate = ist->st->r_frame_rate;

                if (ist && !ost->frame_rate.num) {

                    ost->frame_rate = (AVRational){25, 1};

                    av_log(NULL, AV_LOG_WARNING,

                           "No information "

                           "about the input framerate is available. Falling "

                           "back to a default value of 25fps for output stream #%d:%d. Use the -r option "

                           "if you want a different framerate.\n",

                           ost->file_index, ost->index);

                }

//                    ost->frame_rate = ist->st->avg_frame_rate.num ? ist->st->avg_frame_rate : (AVRational){25, 1};

                if (ost->enc && ost->enc->supported_framerates && !ost->force_fps) {

                    int idx = av_find_nearest_q_idx(ost->frame_rate, ost->enc->supported_framerates);

                    ost->frame_rate = ost->enc->supported_framerates[idx];

                }

                // reduce frame rate for mpeg4 to be within the spec limits

                if (enc_ctx->codec_id == AV_CODEC_ID_MPEG4) {

                    av_reduce(&ost->frame_rate.num, &ost->frame_rate.den,

                              ost->frame_rate.num, ost->frame_rate.den, 65535);

                }

            }



            switch (enc_ctx->codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                enc_ctx->sample_fmt     = ost->filter->filter->inputs[0]->format;

                enc_ctx->sample_rate    = ost->filter->filter->inputs[0]->sample_rate;

                enc_ctx->channel_layout = ost->filter->filter->inputs[0]->channel_layout;

                enc_ctx->channels       = avfilter_link_get_channels(ost->filter->filter->inputs[0]);

                enc_ctx->time_base      = (AVRational){ 1, enc_ctx->sample_rate };

                break;

            case AVMEDIA_TYPE_VIDEO:

                enc_ctx->time_base = av_inv_q(ost->frame_rate);

                if (ost->filter && !(enc_ctx->time_base.num && enc_ctx->time_base.den))

                    enc_ctx->time_base = ost->filter->filter->inputs[0]->time_base;

                if (   av_q2d(enc_ctx->time_base) < 0.001 && video_sync_method != VSYNC_PASSTHROUGH

                   && (video_sync_method == VSYNC_CFR || video_sync_method == VSYNC_VSCFR || (video_sync_method == VSYNC_AUTO && !(oc->oformat->flags & AVFMT_VARIABLE_FPS)))){

                    av_log(oc, AV_LOG_WARNING, "Frame rate very high for a muxer not efficiently supporting it.\n"

                                               "Please consider specifying a lower framerate, a different muxer or -vsync 2\n");

                }

                for (j = 0; j < ost->forced_kf_count; j++)

                    ost->forced_kf_pts[j] = av_rescale_q(ost->forced_kf_pts[j],

                                                         AV_TIME_BASE_Q,

                                                         enc_ctx->time_base);



                enc_ctx->width  = ost->filter->filter->inputs[0]->w;

                enc_ctx->height = ost->filter->filter->inputs[0]->h;

                enc_ctx->sample_aspect_ratio = ost->st->sample_aspect_ratio =

                    ost->frame_aspect_ratio.num ? // overridden by the -aspect cli option

                    av_mul_q(ost->frame_aspect_ratio, (AVRational){ enc_ctx->height, enc_ctx->width }) :

                    ost->filter->filter->inputs[0]->sample_aspect_ratio;

                if (!strncmp(ost->enc->name, "libx264", 7) &&

                    enc_ctx->pix_fmt == AV_PIX_FMT_NONE &&

                    ost->filter->filter->inputs[0]->format != AV_PIX_FMT_YUV420P)

                    av_log(NULL, AV_LOG_WARNING,

                           "No pixel format specified, %s for H.264 encoding chosen.\n"

                           "Use -pix_fmt yuv420p for compatibility with outdated media players.\n",

                           av_get_pix_fmt_name(ost->filter->filter->inputs[0]->format));

                if (!strncmp(ost->enc->name, "mpeg2video", 10) &&

                    enc_ctx->pix_fmt == AV_PIX_FMT_NONE &&

                    ost->filter->filter->inputs[0]->format != AV_PIX_FMT_YUV420P)

                    av_log(NULL, AV_LOG_WARNING,

                           "No pixel format specified, %s for MPEG-2 encoding chosen.\n"

                           "Use -pix_fmt yuv420p for compatibility with outdated media players.\n",

                           av_get_pix_fmt_name(ost->filter->filter->inputs[0]->format));

                enc_ctx->pix_fmt = ost->filter->filter->inputs[0]->format;



                ost->st->avg_frame_rate = ost->frame_rate;



                if (!dec_ctx ||

                    enc_ctx->width   != dec_ctx->width  ||

                    enc_ctx->height  != dec_ctx->height ||

                    enc_ctx->pix_fmt != dec_ctx->pix_fmt) {

                    enc_ctx->bits_per_raw_sample = frame_bits_per_raw_sample;

                }



                if (ost->forced_keyframes) {

                    if (!strncmp(ost->forced_keyframes, "expr:", 5)) {

                        ret = av_expr_parse(&ost->forced_keyframes_pexpr, ost->forced_keyframes+5,

                                            forced_keyframes_const_names, NULL, NULL, NULL, NULL, 0, NULL);

                        if (ret < 0) {

                            av_log(NULL, AV_LOG_ERROR,

                                   "Invalid force_key_frames expression '%s'\n", ost->forced_keyframes+5);

                            return ret;

                        }

                        ost->forced_keyframes_expr_const_values[FKF_N] = 0;

                        ost->forced_keyframes_expr_const_values[FKF_N_FORCED] = 0;

                        ost->forced_keyframes_expr_const_values[FKF_PREV_FORCED_N] = NAN;

                        ost->forced_keyframes_expr_const_values[FKF_PREV_FORCED_T] = NAN;

                    } else {

                        parse_forced_key_frames(ost->forced_keyframes, ost, ost->enc_ctx);

                    }

                }

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                enc_ctx->time_base = (AVRational){1, 1000};

                if (!enc_ctx->width) {

                    enc_ctx->width     = input_streams[ost->source_index]->st->codec->width;

                    enc_ctx->height    = input_streams[ost->source_index]->st->codec->height;

                }

                break;

            case AVMEDIA_TYPE_DATA:

                break;

            default:

                abort();

                break;

            }

            /* two pass mode */

            if (enc_ctx->flags & (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2)) {

                char logfilename[1024];

                FILE *f;



                snprintf(logfilename, sizeof(logfilename), "%s-%d.log",

                         ost->logfile_prefix ? ost->logfile_prefix :

                                               DEFAULT_PASS_LOGFILENAME_PREFIX,

                         i);

                if (!strcmp(ost->enc->name, "libx264")) {

                    av_dict_set(&ost->encoder_opts, "stats", logfilename, AV_DICT_DONT_OVERWRITE);

                } else {

                    if (enc_ctx->flags & CODEC_FLAG_PASS2) {

                        char  *logbuffer;

                        size_t logbuffer_size;

                        if (cmdutils_read_file(logfilename, &logbuffer, &logbuffer_size) < 0) {

                            av_log(NULL, AV_LOG_FATAL, "Error reading log file '%s' for pass-2 encoding\n",

                                   logfilename);

                            exit_program(1);

                        }

                        enc_ctx->stats_in = logbuffer;

                    }

                    if (enc_ctx->flags & CODEC_FLAG_PASS1) {

                        f = av_fopen_utf8(logfilename, "wb");

                        if (!f) {

                            av_log(NULL, AV_LOG_FATAL, "Cannot write log file '%s' for pass-1 encoding: %s\n",

                                logfilename, strerror(errno));

                            exit_program(1);

                        }

                        ost->logfile = f;

                    }

                }

            }

        }



        if (ost->disposition) {

            static const AVOption opts[] = {

                { "disposition"         , NULL, 0, AV_OPT_TYPE_FLAGS, { .i64 = 0 }, INT64_MIN, INT64_MAX, .unit = "flags" },

                { "default"             , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_DEFAULT           },    .unit = "flags" },

                { "dub"                 , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_DUB               },    .unit = "flags" },

                { "original"            , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_ORIGINAL          },    .unit = "flags" },

                { "comment"             , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_COMMENT           },    .unit = "flags" },

                { "lyrics"              , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_LYRICS            },    .unit = "flags" },

                { "karaoke"             , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_KARAOKE           },    .unit = "flags" },

                { "forced"              , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_FORCED            },    .unit = "flags" },

                { "hearing_impaired"    , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_HEARING_IMPAIRED  },    .unit = "flags" },

                { "visual_impaired"     , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_VISUAL_IMPAIRED   },    .unit = "flags" },

                { "clean_effects"       , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_CLEAN_EFFECTS     },    .unit = "flags" },

                { "captions"            , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_CAPTIONS          },    .unit = "flags" },

                { "descriptions"        , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_DESCRIPTIONS      },    .unit = "flags" },

                { "metadata"            , NULL, 0, AV_OPT_TYPE_CONST, { .i64 = AV_DISPOSITION_METADATA          },    .unit = "flags" },

                { NULL },

            };

            static const AVClass class = {

                .class_name = "",

                .item_name  = av_default_item_name,

                .option     = opts,

                .version    = LIBAVUTIL_VERSION_INT,

            };

            const AVClass *pclass = &class;



            ret = av_opt_eval_flags(&pclass, &opts[0], ost->disposition, &ost->st->disposition);

            if (ret < 0)

                goto dump_format;

        }

    }



    /* open each encoder */

    for (i = 0; i < nb_output_streams; i++) {

        ost = output_streams[i];

        if (ost->encoding_needed) {

            AVCodec      *codec = ost->enc;

            AVCodecContext *dec = NULL;



            if ((ist = get_input_stream(ost)))

                dec = ist->dec_ctx;

            if (dec && dec->subtitle_header) {

                /* ASS code assumes this buffer is null terminated so add extra byte. */

                ost->enc_ctx->subtitle_header = av_mallocz(dec->subtitle_header_size + 1);

                if (!ost->enc_ctx->subtitle_header) {

                    ret = AVERROR(ENOMEM);

                    goto dump_format;

                }

                memcpy(ost->enc_ctx->subtitle_header, dec->subtitle_header, dec->subtitle_header_size);

                ost->enc_ctx->subtitle_header_size = dec->subtitle_header_size;

            }

            if (!av_dict_get(ost->encoder_opts, "threads", NULL, 0))

                av_dict_set(&ost->encoder_opts, "threads", "auto", 0);

            av_dict_set(&ost->encoder_opts, "side_data_only_packets", "1", 0);



            if ((ret = avcodec_open2(ost->enc_ctx, codec, &ost->encoder_opts)) < 0) {

                if (ret == AVERROR_EXPERIMENTAL)

                    abort_codec_experimental(codec, 1);

                snprintf(error, sizeof(error), "Error while opening encoder for output stream #%d:%d - maybe incorrect parameters such as bit_rate, rate, width or height",

                        ost->file_index, ost->index);

                goto dump_format;

            }

            if (ost->enc->type == AVMEDIA_TYPE_AUDIO &&

                !(ost->enc->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE))

                av_buffersink_set_frame_size(ost->filter->filter,

                                             ost->enc_ctx->frame_size);

            assert_avoptions(ost->encoder_opts);

            if (ost->enc_ctx->bit_rate && ost->enc_ctx->bit_rate < 1000)

                av_log(NULL, AV_LOG_WARNING, "The bitrate parameter is set too low."

                                             " It takes bits/s as argument, not kbits/s\n");



            ret = avcodec_copy_context(ost->st->codec, ost->enc_ctx);

            if (ret < 0) {

                av_log(NULL, AV_LOG_FATAL,

                       "Error initializing the output stream codec context.\n");

                exit_program(1);

            }



            // copy timebase while removing common factors

            ost->st->time_base = av_add_q(ost->enc_ctx->time_base, (AVRational){0, 1});

            ost->st->codec->codec= ost->enc_ctx->codec;

        } else {

            ret = av_opt_set_dict(ost->enc_ctx, &ost->encoder_opts);

            if (ret < 0) {

                av_log(NULL, AV_LOG_FATAL,

                    "Error setting up codec context options.\n");

                return ret;

            }

            // copy timebase while removing common factors

            ost->st->time_base = av_add_q(ost->st->codec->time_base, (AVRational){0, 1});

        }

    }



    /* init input streams */

    for (i = 0; i < nb_input_streams; i++)

        if ((ret = init_input_stream(i, error, sizeof(error))) < 0) {

            for (i = 0; i < nb_output_streams; i++) {

                ost = output_streams[i];

                avcodec_close(ost->enc_ctx);

            }

            goto dump_format;

        }



    /* discard unused programs */

    for (i = 0; i < nb_input_files; i++) {

        InputFile *ifile = input_files[i];

        for (j = 0; j < ifile->ctx->nb_programs; j++) {

            AVProgram *p = ifile->ctx->programs[j];

            int discard  = AVDISCARD_ALL;



            for (k = 0; k < p->nb_stream_indexes; k++)

                if (!input_streams[ifile->ist_index + p->stream_index[k]]->discard) {

                    discard = AVDISCARD_DEFAULT;

                    break;

                }

            p->discard = discard;

        }

    }



    /* open files and write file headers */

    for (i = 0; i < nb_output_files; i++) {

        oc = output_files[i]->ctx;

        oc->interrupt_callback = int_cb;

        if ((ret = avformat_write_header(oc, &output_files[i]->opts)) < 0) {

            snprintf(error, sizeof(error),

                     "Could not write header for output file #%d "

                     "(incorrect codec parameters ?): %s",

                     i, av_err2str(ret));

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

//         assert_avoptions(output_files[i]->opts);

        if (strcmp(oc->oformat->name, "rtp")) {

            want_sdp = 0;

        }

    }



 dump_format:

    /* dump the file output parameters - cannot be done before in case

       of stream copy */

    for (i = 0; i < nb_output_files; i++) {

        av_dump_format(output_files[i]->ctx, i, output_files[i]->ctx->filename, 1);

    }



    /* dump the stream mapping */

    av_log(NULL, AV_LOG_INFO, "Stream mapping:\n");

    for (i = 0; i < nb_input_streams; i++) {

        ist = input_streams[i];



        for (j = 0; j < ist->nb_filters; j++) {

            if (ist->filters[j]->graph->graph_desc) {

                av_log(NULL, AV_LOG_INFO, "  Stream #%d:%d (%s) -> %s",

                       ist->file_index, ist->st->index, ist->dec ? ist->dec->name : "?",

                       ist->filters[j]->name);

                if (nb_filtergraphs > 1)

                    av_log(NULL, AV_LOG_INFO, " (graph %d)", ist->filters[j]->graph->index);

                av_log(NULL, AV_LOG_INFO, "\n");

            }

        }

    }



    for (i = 0; i < nb_output_streams; i++) {

        ost = output_streams[i];



        if (ost->attachment_filename) {

            /* an attached file */

            av_log(NULL, AV_LOG_INFO, "  File %s -> Stream #%d:%d\n",

                   ost->attachment_filename, ost->file_index, ost->index);

            continue;

        }



        if (ost->filter && ost->filter->graph->graph_desc) {

            /* output from a complex graph */

            av_log(NULL, AV_LOG_INFO, "  %s", ost->filter->name);

            if (nb_filtergraphs > 1)

                av_log(NULL, AV_LOG_INFO, " (graph %d)", ost->filter->graph->index);



            av_log(NULL, AV_LOG_INFO, " -> Stream #%d:%d (%s)\n", ost->file_index,

                   ost->index, ost->enc ? ost->enc->name : "?");

            continue;

        }



        av_log(NULL, AV_LOG_INFO, "  Stream #%d:%d -> #%d:%d",

               input_streams[ost->source_index]->file_index,

               input_streams[ost->source_index]->st->index,

               ost->file_index,

               ost->index);

        if (ost->sync_ist != input_streams[ost->source_index])

            av_log(NULL, AV_LOG_INFO, " [sync #%d:%d]",

                   ost->sync_ist->file_index,

                   ost->sync_ist->st->index);

        if (ost->stream_copy)

            av_log(NULL, AV_LOG_INFO, " (copy)");

        else {

            const AVCodec *in_codec    = input_streams[ost->source_index]->dec;

            const AVCodec *out_codec   = ost->enc;

            const char *decoder_name   = "?";

            const char *in_codec_name  = "?";

            const char *encoder_name   = "?";

            const char *out_codec_name = "?";

            const AVCodecDescriptor *desc;



            if (in_codec) {

                decoder_name  = in_codec->name;

                desc = avcodec_descriptor_get(in_codec->id);

                if (desc)

                    in_codec_name = desc->name;

                if (!strcmp(decoder_name, in_codec_name))

                    decoder_name = "native";

            }



            if (out_codec) {

                encoder_name   = out_codec->name;

                desc = avcodec_descriptor_get(out_codec->id);

                if (desc)

                    out_codec_name = desc->name;

                if (!strcmp(encoder_name, out_codec_name))

                    encoder_name = "native";

            }



            av_log(NULL, AV_LOG_INFO, " (%s (%s) -> %s (%s))",

                   in_codec_name, decoder_name,

                   out_codec_name, encoder_name);

        }

        av_log(NULL, AV_LOG_INFO, "\n");

    }



    if (ret) {

        av_log(NULL, AV_LOG_ERROR, "%s\n", error);

        return ret;

    }



    if (sdp_filename || want_sdp) {

        print_sdp();

    }



    transcode_init_done = 1;



    return 0;

}
