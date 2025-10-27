static void write_frame(AVFormatContext *s, AVPacket *pkt, OutputStream *ost)

{

    AVBitStreamFilterContext *bsfc = ost->bitstream_filters;

    AVCodecContext          *avctx = ost->encoding_needed ? ost->enc_ctx : ost->st->codec;

    int ret;



    if (!ost->st->codec->extradata_size && ost->enc_ctx->extradata_size) {

        ost->st->codec->extradata = av_mallocz(ost->enc_ctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);

        if (ost->st->codec->extradata) {

            memcpy(ost->st->codec->extradata, ost->enc_ctx->extradata, ost->enc_ctx->extradata_size);

            ost->st->codec->extradata_size = ost->enc_ctx->extradata_size;

        }

    }



    if ((avctx->codec_type == AVMEDIA_TYPE_VIDEO && video_sync_method == VSYNC_DROP) ||

        (avctx->codec_type == AVMEDIA_TYPE_AUDIO && audio_sync_method < 0))

        pkt->pts = pkt->dts = AV_NOPTS_VALUE;



    /*

     * Audio encoders may split the packets --  #frames in != #packets out.

     * But there is no reordering, so we can limit the number of output packets

     * by simply dropping them here.

     * Counting encoded video frames needs to be done separately because of

     * reordering, see do_video_out()

     */

    if (!(avctx->codec_type == AVMEDIA_TYPE_VIDEO && avctx->codec)) {

        if (ost->frame_number >= ost->max_frames) {

            av_free_packet(pkt);

            return;

        }

        ost->frame_number++;

    }

    if (avctx->codec_type == AVMEDIA_TYPE_VIDEO) {

        int i;

        uint8_t *sd = av_packet_get_side_data(pkt, AV_PKT_DATA_QUALITY_STATS,

                                              NULL);

        ost->quality = sd ? AV_RL32(sd) : -1;

        ost->pict_type = sd ? sd[4] : AV_PICTURE_TYPE_NONE;



        for (i = 0; i<FF_ARRAY_ELEMS(ost->error); i++) {

            if (sd && i < sd[5])

                ost->error[i] = AV_RL64(sd + 8 + 8*i);

            else

                ost->error[i] = -1;

        }

    }



    if (bsfc)

        av_packet_split_side_data(pkt);



    while (bsfc) {

        AVPacket new_pkt = *pkt;

        AVDictionaryEntry *bsf_arg = av_dict_get(ost->bsf_args,

                                                 bsfc->filter->name,

                                                 NULL, 0);

        int a = av_bitstream_filter_filter(bsfc, avctx,

                                           bsf_arg ? bsf_arg->value : NULL,

                                           &new_pkt.data, &new_pkt.size,

                                           pkt->data, pkt->size,

                                           pkt->flags & AV_PKT_FLAG_KEY);

        if(a == 0 && new_pkt.data != pkt->data && new_pkt.destruct) {

            uint8_t *t = av_malloc(new_pkt.size + AV_INPUT_BUFFER_PADDING_SIZE); //the new should be a subset of the old so cannot overflow

            if(t) {

                memcpy(t, new_pkt.data, new_pkt.size);

                memset(t + new_pkt.size, 0, AV_INPUT_BUFFER_PADDING_SIZE);

                new_pkt.data = t;

                new_pkt.buf = NULL;

                a = 1;

            } else

                a = AVERROR(ENOMEM);

        }

        if (a > 0) {

            pkt->side_data = NULL;

            pkt->side_data_elems = 0;

            av_free_packet(pkt);

            new_pkt.buf = av_buffer_create(new_pkt.data, new_pkt.size,

                                           av_buffer_default_free, NULL, 0);

            if (!new_pkt.buf)

                exit_program(1);

        } else if (a < 0) {

            new_pkt = *pkt;

            av_log(NULL, AV_LOG_ERROR, "Failed to open bitstream filter %s for stream %d with codec %s",

                   bsfc->filter->name, pkt->stream_index,

                   avctx->codec ? avctx->codec->name : "copy");

            print_error("", a);

            if (exit_on_error)

                exit_program(1);

        }

        *pkt = new_pkt;



        bsfc = bsfc->next;

    }



    if (!(s->oformat->flags & AVFMT_NOTIMESTAMPS)) {

        if (pkt->dts != AV_NOPTS_VALUE &&

            pkt->pts != AV_NOPTS_VALUE &&

            pkt->dts > pkt->pts) {

            av_log(s, AV_LOG_WARNING, "Invalid DTS: %"PRId64" PTS: %"PRId64" in output stream %d:%d, replacing by guess\n",

                   pkt->dts, pkt->pts,

                   ost->file_index, ost->st->index);

            pkt->pts =

            pkt->dts = pkt->pts + pkt->dts + ost->last_mux_dts + 1

                     - FFMIN3(pkt->pts, pkt->dts, ost->last_mux_dts + 1)

                     - FFMAX3(pkt->pts, pkt->dts, ost->last_mux_dts + 1);

        }

     if(

        (avctx->codec_type == AVMEDIA_TYPE_AUDIO || avctx->codec_type == AVMEDIA_TYPE_VIDEO) &&

        pkt->dts != AV_NOPTS_VALUE &&

        ost->last_mux_dts != AV_NOPTS_VALUE) {

      int64_t max = ost->last_mux_dts + !(s->oformat->flags & AVFMT_TS_NONSTRICT);

      if (pkt->dts < max) {

        int loglevel = max - pkt->dts > 2 || avctx->codec_type == AVMEDIA_TYPE_VIDEO ? AV_LOG_WARNING : AV_LOG_DEBUG;

        av_log(s, loglevel, "Non-monotonous DTS in output stream "

               "%d:%d; previous: %"PRId64", current: %"PRId64"; ",

               ost->file_index, ost->st->index, ost->last_mux_dts, pkt->dts);

        if (exit_on_error) {

            av_log(NULL, AV_LOG_FATAL, "aborting.\n");

            exit_program(1);

        }

        av_log(s, loglevel, "changing to %"PRId64". This may result "

               "in incorrect timestamps in the output file.\n",

               max);

        if(pkt->pts >= pkt->dts)

            pkt->pts = FFMAX(pkt->pts, max);

        pkt->dts = max;

      }

     }

    }

    ost->last_mux_dts = pkt->dts;



    ost->data_size += pkt->size;

    ost->packets_written++;



    pkt->stream_index = ost->index;



    if (debug_ts) {

        av_log(NULL, AV_LOG_INFO, "muxer <- type:%s "

                "pkt_pts:%s pkt_pts_time:%s pkt_dts:%s pkt_dts_time:%s size:%d\n",

                av_get_media_type_string(ost->enc_ctx->codec_type),

                av_ts2str(pkt->pts), av_ts2timestr(pkt->pts, &ost->st->time_base),

                av_ts2str(pkt->dts), av_ts2timestr(pkt->dts, &ost->st->time_base),

                pkt->size

              );

    }



    ret = av_interleaved_write_frame(s, pkt);

    if (ret < 0) {

        print_error("av_interleaved_write_frame()", ret);

        main_return_code = 1;

        close_all_output_streams(ost, MUXER_FINISHED | ENCODER_FINISHED, ENCODER_FINISHED);

    }

    av_free_packet(pkt);

}
