int avcodec_decode_subtitle2(AVCodecContext *avctx, AVSubtitle *sub,

                             int *got_sub_ptr,

                             AVPacket *avpkt)

{

    int i, ret = 0;



    if (!avpkt->data && avpkt->size) {

        av_log(avctx, AV_LOG_ERROR, "invalid packet: NULL data, size != 0\n");

        return AVERROR(EINVAL);

    }

    if (!avctx->codec)

        return AVERROR(EINVAL);

    if (avctx->codec->type != AVMEDIA_TYPE_SUBTITLE) {

        av_log(avctx, AV_LOG_ERROR, "Invalid media type for subtitles\n");

        return AVERROR(EINVAL);

    }



    *got_sub_ptr = 0;

    get_subtitle_defaults(sub);



    if ((avctx->codec->capabilities & AV_CODEC_CAP_DELAY) || avpkt->size) {

        AVPacket pkt_recoded;

        AVPacket tmp = *avpkt;

        int did_split = av_packet_split_side_data(&tmp);

        //apply_param_change(avctx, &tmp);



        if (did_split) {

            /* FFMIN() prevents overflow in case the packet wasn't allocated with

             * proper padding.

             * If the side data is smaller than the buffer padding size, the

             * remaining bytes should have already been filled with zeros by the

             * original packet allocation anyway. */

            memset(tmp.data + tmp.size, 0,

                   FFMIN(avpkt->size - tmp.size, AV_INPUT_BUFFER_PADDING_SIZE));

        }



        pkt_recoded = tmp;

        ret = recode_subtitle(avctx, &pkt_recoded, &tmp);

        if (ret < 0) {

            *got_sub_ptr = 0;

        } else {

            avctx->internal->pkt = &pkt_recoded;



            if (avctx->pkt_timebase.den && avpkt->pts != AV_NOPTS_VALUE)

                sub->pts = av_rescale_q(avpkt->pts,

                                        avctx->pkt_timebase, AV_TIME_BASE_Q);

            ret = avctx->codec->decode(avctx, sub, got_sub_ptr, &pkt_recoded);

            av_assert1((ret >= 0) >= !!*got_sub_ptr &&

                       !!*got_sub_ptr >= !!sub->num_rects);



            if (sub->num_rects && !sub->end_display_time && avpkt->duration &&

                avctx->pkt_timebase.num) {

                AVRational ms = { 1, 1000 };

                sub->end_display_time = av_rescale_q(avpkt->duration,

                                                     avctx->pkt_timebase, ms);

            }



            for (i = 0; i < sub->num_rects; i++) {

                if (sub->rects[i]->ass && !utf8_check(sub->rects[i]->ass)) {

                    av_log(avctx, AV_LOG_ERROR,

                           "Invalid UTF-8 in decoded subtitles text; "

                           "maybe missing -sub_charenc option\n");

                    avsubtitle_free(sub);

                    return AVERROR_INVALIDDATA;

                }

            }



            if (tmp.data != pkt_recoded.data) { // did we recode?

                /* prevent from destroying side data from original packet */

                pkt_recoded.side_data = NULL;

                pkt_recoded.side_data_elems = 0;



                av_packet_unref(&pkt_recoded);

            }

            if (avctx->codec_descriptor->props & AV_CODEC_PROP_BITMAP_SUB)

                sub->format = 0;

            else if (avctx->codec_descriptor->props & AV_CODEC_PROP_TEXT_SUB)

                sub->format = 1;

            avctx->internal->pkt = NULL;

        }



        if (did_split) {

            av_packet_free_side_data(&tmp);

            if(ret == tmp.size)

                ret = avpkt->size;

        }



        if (*got_sub_ptr)

            avctx->frame_number++;

    }



    return ret;

}
