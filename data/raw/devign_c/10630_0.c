static int init_output_stream_encode(OutputStream *ost)

{

    InputStream *ist = get_input_stream(ost);

    AVCodecContext *enc_ctx = ost->enc_ctx;

    AVCodecContext *dec_ctx = NULL;

    AVFormatContext *oc = output_files[ost->file_index]->ctx;

    int j, ret;



    set_encoder_id(output_files[ost->file_index], ost);



    if (ist) {

        ost->st->disposition          = ist->st->disposition;



        dec_ctx = ist->dec_ctx;



        enc_ctx->chroma_sample_location = dec_ctx->chroma_sample_location;

    } else {

        for (j = 0; j < oc->nb_streams; j++) {

            AVStream *st = oc->streams[j];

            if (st != ost->st && st->codecpar->codec_type == ost->st->codecpar->codec_type)

                break;

        }

        if (j == oc->nb_streams)

            if (ost->st->codecpar->codec_type == AVMEDIA_TYPE_AUDIO ||

                ost->st->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)

                ost->st->disposition = AV_DISPOSITION_DEFAULT;

    }



    if ((enc_ctx->codec_type == AVMEDIA_TYPE_VIDEO ||

         enc_ctx->codec_type == AVMEDIA_TYPE_AUDIO) &&

         filtergraph_is_simple(ost->filter->graph)) {

            FilterGraph *fg = ost->filter->graph;



            if (configure_filtergraph(fg)) {

                av_log(NULL, AV_LOG_FATAL, "Error opening filters!\n");

                exit_program(1);

            }

    }



    if (enc_ctx->codec_type == AVMEDIA_TYPE_VIDEO) {

        if (!ost->frame_rate.num)

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

//      ost->frame_rate = ist->st->avg_frame_rate.num ? ist->st->avg_frame_rate : (AVRational){25, 1};

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

        enc_ctx->sample_fmt     = av_buffersink_get_format(ost->filter->filter);

        if (dec_ctx)

            enc_ctx->bits_per_raw_sample = FFMIN(dec_ctx->bits_per_raw_sample,

                                                 av_get_bytes_per_sample(enc_ctx->sample_fmt) << 3);

        enc_ctx->sample_rate    = av_buffersink_get_sample_rate(ost->filter->filter);

        enc_ctx->channel_layout = av_buffersink_get_channel_layout(ost->filter->filter);

        enc_ctx->channels       = av_buffersink_get_channels(ost->filter->filter);

        enc_ctx->time_base      = (AVRational){ 1, enc_ctx->sample_rate };

        break;

    case AVMEDIA_TYPE_VIDEO:

        enc_ctx->time_base = av_inv_q(ost->frame_rate);

        if (!(enc_ctx->time_base.num && enc_ctx->time_base.den))

            enc_ctx->time_base = av_buffersink_get_time_base(ost->filter->filter);

        if (   av_q2d(enc_ctx->time_base) < 0.001 && video_sync_method != VSYNC_PASSTHROUGH

           && (video_sync_method == VSYNC_CFR || video_sync_method == VSYNC_VSCFR || (video_sync_method == VSYNC_AUTO && !(oc->oformat->flags & AVFMT_VARIABLE_FPS)))){

            av_log(oc, AV_LOG_WARNING, "Frame rate very high for a muxer not efficiently supporting it.\n"

                                       "Please consider specifying a lower framerate, a different muxer or -vsync 2\n");

        }

        for (j = 0; j < ost->forced_kf_count; j++)

            ost->forced_kf_pts[j] = av_rescale_q(ost->forced_kf_pts[j],

                                                 AV_TIME_BASE_Q,

                                                 enc_ctx->time_base);



        enc_ctx->width  = av_buffersink_get_w(ost->filter->filter);

        enc_ctx->height = av_buffersink_get_h(ost->filter->filter);

        enc_ctx->sample_aspect_ratio = ost->st->sample_aspect_ratio =

            ost->frame_aspect_ratio.num ? // overridden by the -aspect cli option

            av_mul_q(ost->frame_aspect_ratio, (AVRational){ enc_ctx->height, enc_ctx->width }) :

            av_buffersink_get_sample_aspect_ratio(ost->filter->filter);

        if (!strncmp(ost->enc->name, "libx264", 7) &&

            enc_ctx->pix_fmt == AV_PIX_FMT_NONE &&

            av_buffersink_get_format(ost->filter->filter) != AV_PIX_FMT_YUV420P)

            av_log(NULL, AV_LOG_WARNING,

                   "No pixel format specified, %s for H.264 encoding chosen.\n"

                   "Use -pix_fmt yuv420p for compatibility with outdated media players.\n",

                   av_get_pix_fmt_name(av_buffersink_get_format(ost->filter->filter)));

        if (!strncmp(ost->enc->name, "mpeg2video", 10) &&

            enc_ctx->pix_fmt == AV_PIX_FMT_NONE &&

            av_buffersink_get_format(ost->filter->filter) != AV_PIX_FMT_YUV420P)

            av_log(NULL, AV_LOG_WARNING,

                   "No pixel format specified, %s for MPEG-2 encoding chosen.\n"

                   "Use -pix_fmt yuv420p for compatibility with outdated media players.\n",

                   av_get_pix_fmt_name(av_buffersink_get_format(ost->filter->filter)));

        enc_ctx->pix_fmt = av_buffersink_get_format(ost->filter->filter);

        if (dec_ctx)

            enc_ctx->bits_per_raw_sample = FFMIN(dec_ctx->bits_per_raw_sample,

                                                 av_pix_fmt_desc_get(enc_ctx->pix_fmt)->comp[0].depth);



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



            // Don't parse the 'forced_keyframes' in case of 'keep-source-keyframes',

            // parse it only for static kf timings

            } else if(strncmp(ost->forced_keyframes, "source", 6)) {

                parse_forced_key_frames(ost->forced_keyframes, ost, ost->enc_ctx);

            }

        }

        break;

    case AVMEDIA_TYPE_SUBTITLE:

        enc_ctx->time_base = (AVRational){1, 1000};

        if (!enc_ctx->width) {

            enc_ctx->width     = input_streams[ost->source_index]->st->codecpar->width;

            enc_ctx->height    = input_streams[ost->source_index]->st->codecpar->height;

        }

        break;

    case AVMEDIA_TYPE_DATA:

        break;

    default:

        abort();

        break;

    }



    return 0;

}
