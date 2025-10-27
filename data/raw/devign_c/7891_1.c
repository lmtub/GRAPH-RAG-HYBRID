static int decode_simple_internal(AVCodecContext *avctx, AVFrame *frame)

{

    AVCodecInternal   *avci = avctx->internal;

    DecodeSimpleContext *ds = &avci->ds;

    AVPacket           *pkt = ds->in_pkt;

    // copy to ensure we do not change pkt

    AVPacket tmp;

    int got_frame, actual_got_frame, did_split;

    int ret;



    if (!pkt->data && !avci->draining) {

        av_packet_unref(pkt);

        ret = ff_decode_get_packet(avctx, pkt);

        if (ret < 0 && ret != AVERROR_EOF)

            return ret;

    }



    // Some codecs (at least wma lossless) will crash when feeding drain packets

    // after EOF was signaled.

    if (avci->draining_done)

        return AVERROR_EOF;



    if (!pkt->data &&

        !(avctx->codec->capabilities & AV_CODEC_CAP_DELAY ||

          avctx->active_thread_type & FF_THREAD_FRAME))

        return AVERROR_EOF;



    tmp = *pkt;

#if FF_API_MERGE_SD

FF_DISABLE_DEPRECATION_WARNINGS

    did_split = av_packet_split_side_data(&tmp);



    if (did_split) {

        ret = extract_packet_props(avctx->internal, &tmp);

        if (ret < 0)

            return ret;



        ret = apply_param_change(avctx, &tmp);

        if (ret < 0)

            return ret;

    }

FF_ENABLE_DEPRECATION_WARNINGS

#endif



    got_frame = 0;



    if (HAVE_THREADS && avctx->active_thread_type & FF_THREAD_FRAME) {

        ret = ff_thread_decode_frame(avctx, frame, &got_frame, &tmp);

    } else {

        ret = avctx->codec->decode(avctx, frame, &got_frame, &tmp);



        if (!(avctx->codec->caps_internal & FF_CODEC_CAP_SETS_PKT_DTS))

            frame->pkt_dts = pkt->dts;

        if (avctx->codec->type == AVMEDIA_TYPE_VIDEO) {

            if(!avctx->has_b_frames)

                frame->pkt_pos = pkt->pos;

            //FIXME these should be under if(!avctx->has_b_frames)

            /* get_buffer is supposed to set frame parameters */

            if (!(avctx->codec->capabilities & AV_CODEC_CAP_DR1)) {

                if (!frame->sample_aspect_ratio.num)  frame->sample_aspect_ratio = avctx->sample_aspect_ratio;

                if (!frame->width)                    frame->width               = avctx->width;

                if (!frame->height)                   frame->height              = avctx->height;

                if (frame->format == AV_PIX_FMT_NONE) frame->format              = avctx->pix_fmt;

            }

        }

    }

    emms_c();

    actual_got_frame = got_frame;



    if (avctx->codec->type == AVMEDIA_TYPE_VIDEO) {

        if (frame->flags & AV_FRAME_FLAG_DISCARD)

            got_frame = 0;

        if (got_frame)

            frame->best_effort_timestamp = guess_correct_pts(avctx,

                                                             frame->pts,

                                                             frame->pkt_dts);

    } else if (avctx->codec->type == AVMEDIA_TYPE_AUDIO) {

        uint8_t *side;

        int side_size;

        uint32_t discard_padding = 0;

        uint8_t skip_reason = 0;

        uint8_t discard_reason = 0;



        if (ret >= 0 && got_frame) {

            frame->best_effort_timestamp = guess_correct_pts(avctx,

                                                             frame->pts,

                                                             frame->pkt_dts);

            if (frame->format == AV_SAMPLE_FMT_NONE)

                frame->format = avctx->sample_fmt;

            if (!frame->channel_layout)

                frame->channel_layout = avctx->channel_layout;

            if (!frame->channels)

                frame->channels = avctx->channels;

            if (!frame->sample_rate)

                frame->sample_rate = avctx->sample_rate;

        }



        side= av_packet_get_side_data(pkt, AV_PKT_DATA_SKIP_SAMPLES, &side_size);

        if(side && side_size>=10) {

            avctx->internal->skip_samples = AV_RL32(side) * avctx->internal->skip_samples_multiplier;

            discard_padding = AV_RL32(side + 4);

            av_log(avctx, AV_LOG_DEBUG, "skip %d / discard %d samples due to side data\n",

                   avctx->internal->skip_samples, (int)discard_padding);

            skip_reason = AV_RL8(side + 8);

            discard_reason = AV_RL8(side + 9);

        }



        if ((frame->flags & AV_FRAME_FLAG_DISCARD) && got_frame &&

            !(avctx->flags2 & AV_CODEC_FLAG2_SKIP_MANUAL)) {

            avctx->internal->skip_samples = FFMAX(0, avctx->internal->skip_samples - frame->nb_samples);

            got_frame = 0;

        }



        if (avctx->internal->skip_samples > 0 && got_frame &&

            !(avctx->flags2 & AV_CODEC_FLAG2_SKIP_MANUAL)) {

            if(frame->nb_samples <= avctx->internal->skip_samples){

                got_frame = 0;

                avctx->internal->skip_samples -= frame->nb_samples;

                av_log(avctx, AV_LOG_DEBUG, "skip whole frame, skip left: %d\n",

                       avctx->internal->skip_samples);

            } else {

                av_samples_copy(frame->extended_data, frame->extended_data, 0, avctx->internal->skip_samples,

                                frame->nb_samples - avctx->internal->skip_samples, avctx->channels, frame->format);

                if(avctx->pkt_timebase.num && avctx->sample_rate) {

                    int64_t diff_ts = av_rescale_q(avctx->internal->skip_samples,

                                                   (AVRational){1, avctx->sample_rate},

                                                   avctx->pkt_timebase);

                    if(frame->pts!=AV_NOPTS_VALUE)

                        frame->pts += diff_ts;

#if FF_API_PKT_PTS

FF_DISABLE_DEPRECATION_WARNINGS

                    if(frame->pkt_pts!=AV_NOPTS_VALUE)

                        frame->pkt_pts += diff_ts;

FF_ENABLE_DEPRECATION_WARNINGS

#endif

                    if(frame->pkt_dts!=AV_NOPTS_VALUE)

                        frame->pkt_dts += diff_ts;

                    if (frame->pkt_duration >= diff_ts)

                        frame->pkt_duration -= diff_ts;

                } else {

                    av_log(avctx, AV_LOG_WARNING, "Could not update timestamps for skipped samples.\n");

                }

                av_log(avctx, AV_LOG_DEBUG, "skip %d/%d samples\n",

                       avctx->internal->skip_samples, frame->nb_samples);

                frame->nb_samples -= avctx->internal->skip_samples;

                avctx->internal->skip_samples = 0;

            }

        }



        if (discard_padding > 0 && discard_padding <= frame->nb_samples && got_frame &&

            !(avctx->flags2 & AV_CODEC_FLAG2_SKIP_MANUAL)) {

            if (discard_padding == frame->nb_samples) {

                got_frame = 0;

            } else {

                if(avctx->pkt_timebase.num && avctx->sample_rate) {

                    int64_t diff_ts = av_rescale_q(frame->nb_samples - discard_padding,

                                                   (AVRational){1, avctx->sample_rate},

                                                   avctx->pkt_timebase);

                    frame->pkt_duration = diff_ts;

                } else {

                    av_log(avctx, AV_LOG_WARNING, "Could not update timestamps for discarded samples.\n");

                }

                av_log(avctx, AV_LOG_DEBUG, "discard %d/%d samples\n",

                       (int)discard_padding, frame->nb_samples);

                frame->nb_samples -= discard_padding;

            }

        }



        if ((avctx->flags2 & AV_CODEC_FLAG2_SKIP_MANUAL) && got_frame) {

            AVFrameSideData *fside = av_frame_new_side_data(frame, AV_FRAME_DATA_SKIP_SAMPLES, 10);

            if (fside) {

                AV_WL32(fside->data, avctx->internal->skip_samples);

                AV_WL32(fside->data + 4, discard_padding);

                AV_WL8(fside->data + 8, skip_reason);

                AV_WL8(fside->data + 9, discard_reason);

                avctx->internal->skip_samples = 0;

            }

        }

    }

#if FF_API_MERGE_SD

    if (did_split) {

        av_packet_free_side_data(&tmp);

        if(ret == tmp.size)

            ret = pkt->size;

    }

#endif



    if (avctx->codec->type == AVMEDIA_TYPE_AUDIO &&

        !avci->showed_multi_packet_warning &&

        ret >= 0 && ret != pkt->size && !(avctx->codec->capabilities & AV_CODEC_CAP_SUBFRAMES)) {

        av_log(avctx, AV_LOG_WARNING, "Multiple frames in a packet.\n");

        avci->showed_multi_packet_warning = 1;

    }



    if (!got_frame)

        av_frame_unref(frame);



    if (ret >= 0 && avctx->codec->type == AVMEDIA_TYPE_VIDEO && !(avctx->flags & AV_CODEC_FLAG_TRUNCATED))

        ret = pkt->size;



#if FF_API_AVCTX_TIMEBASE

    if (avctx->framerate.num > 0 && avctx->framerate.den > 0)

        avctx->time_base = av_inv_q(av_mul_q(avctx->framerate, (AVRational){avctx->ticks_per_frame, 1}));

#endif



    /* do not stop draining when actual_got_frame != 0 or ret < 0 */

    /* got_frame == 0 but actual_got_frame != 0 when frame is discarded */

    if (avctx->internal->draining && !actual_got_frame) {

        if (ret < 0) {

            /* prevent infinite loop if a decoder wrongly always return error on draining */

            /* reasonable nb_errors_max = maximum b frames + thread count */

            int nb_errors_max = 20 + (HAVE_THREADS && avctx->active_thread_type & FF_THREAD_FRAME ?

                                avctx->thread_count : 1);



            if (avci->nb_draining_errors++ >= nb_errors_max) {

                av_log(avctx, AV_LOG_ERROR, "Too many errors when draining, this is a bug. "

                       "Stop draining and force EOF.\n");

                avci->draining_done = 1;

                ret = AVERROR_BUG;

            }

        } else {

            avci->draining_done = 1;

        }

    }



    avci->compat_decode_consumed += ret;



    if (ret >= pkt->size || ret < 0) {

        av_packet_unref(pkt);

    } else {

        int consumed = ret;



        pkt->data                += consumed;

        pkt->size                -= consumed;

        avci->last_pkt_props->size -= consumed; // See extract_packet_props() comment.

        pkt->pts                  = AV_NOPTS_VALUE;

        pkt->dts                  = AV_NOPTS_VALUE;

        avci->last_pkt_props->pts = AV_NOPTS_VALUE;

        avci->last_pkt_props->dts = AV_NOPTS_VALUE;

    }



    if (got_frame)

        av_assert0(frame->buf[0]);



    return ret < 0 ? ret : 0;

}
