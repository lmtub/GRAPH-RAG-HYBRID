static int subviewer_decode_frame(AVCodecContext *avctx,

                                  void *data, int *got_sub_ptr, AVPacket *avpkt)

{

    char c;

    AVSubtitle *sub = data;

    const char *ptr = avpkt->data;

    AVBPrint buf;



    /* To be removed later */

    if (sscanf(ptr, "%*u:%*u:%*u.%*u,%*u:%*u:%*u.%*u%c", &c) == 1) {

        av_log(avctx, AV_LOG_ERROR, "AVPacket is not clean (contains timing "

               "information). You need to upgrade your libavformat or "

               "sanitize your packet.\n");

        return AVERROR_INVALIDDATA;

    }



    av_bprint_init(&buf, 0, AV_BPRINT_SIZE_UNLIMITED);

    // note: no need to rescale pts & duration since they are in the same

    // timebase as ASS (1/100)

    if (ptr && avpkt->size > 0 && !subviewer_event_to_ass(&buf, ptr))

        ff_ass_add_rect(sub, buf.str, avpkt->pts, avpkt->duration, 0);

    *got_sub_ptr = sub->num_rects > 0;

    av_bprint_finalize(&buf, NULL);

    return avpkt->size;

}
