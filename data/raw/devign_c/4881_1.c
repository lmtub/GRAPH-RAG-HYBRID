static int compat_decode(AVCodecContext *avctx, AVFrame *frame,

                         int *got_frame, AVPacket *pkt)

{

    AVCodecInternal *avci = avctx->internal;

    int ret;



    av_assert0(avci->compat_decode_consumed == 0);



    *got_frame = 0;

    avci->compat_decode = 1;



    if (avci->compat_decode_partial_size > 0 &&

        avci->compat_decode_partial_size != pkt->size) {

        av_log(avctx, AV_LOG_ERROR,

               "Got unexpected packet size after a partial decode\n");

        ret = AVERROR(EINVAL);

        goto finish;

    }



    if (!avci->compat_decode_partial_size) {

        ret = avcodec_send_packet(avctx, pkt);

        if (ret == AVERROR_EOF)

            ret = 0;

        else if (ret == AVERROR(EAGAIN)) {

            /* we fully drain all the output in each decode call, so this should not

             * ever happen */

            ret = AVERROR_BUG;

            goto finish;

        } else if (ret < 0)

            goto finish;

    }



    while (ret >= 0) {

        ret = avcodec_receive_frame(avctx, frame);

        if (ret < 0) {

            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)

                ret = 0;

            goto finish;

        }



        if (frame != avci->compat_decode_frame) {

            if (!avctx->refcounted_frames) {

                ret = unrefcount_frame(avci, frame);

                if (ret < 0)

                    goto finish;

            }



            *got_frame = 1;

            frame = avci->compat_decode_frame;

        } else {

            if (!avci->compat_decode_warned) {

                av_log(avctx, AV_LOG_WARNING, "The deprecated avcodec_decode_* "

                       "API cannot return all the frames for this decoder. "

                       "Some frames will be dropped. Update your code to the "

                       "new decoding API to fix this.\n");

                avci->compat_decode_warned = 1;

            }

        }



        if (avci->draining || (!avctx->codec->bsfs && avci->compat_decode_consumed < pkt->size))

            break;

    }



finish:

    if (ret == 0) {

        /* if there are any bsfs then assume full packet is always consumed */

        if (avctx->codec->bsfs)

            ret = pkt->size;

        else

            ret = FFMIN(avci->compat_decode_consumed, pkt->size);

    }

    avci->compat_decode_consumed = 0;

    avci->compat_decode_partial_size = (ret >= 0) ? pkt->size - ret : 0;



    return ret;

}
