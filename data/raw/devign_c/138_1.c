void ff_print_debug_info(MpegEncContext *s, Picture *p)

{

    AVFrame *pict;

    if (s->avctx->hwaccel || !p || !p->mb_type)

        return;

    pict = &p->f;



    if (s->avctx->debug & (FF_DEBUG_SKIP | FF_DEBUG_QP | FF_DEBUG_MB_TYPE)) {

        int x,y;



        av_log(s->avctx,AV_LOG_DEBUG,"New frame, type: ");

        switch (pict->pict_type) {

        case AV_PICTURE_TYPE_I:

            av_log(s->avctx,AV_LOG_DEBUG,"I\n");

            break;

        case AV_PICTURE_TYPE_P:

            av_log(s->avctx,AV_LOG_DEBUG,"P\n");

            break;

        case AV_PICTURE_TYPE_B:

            av_log(s->avctx,AV_LOG_DEBUG,"B\n");

            break;

        case AV_PICTURE_TYPE_S:

            av_log(s->avctx,AV_LOG_DEBUG,"S\n");

            break;

        case AV_PICTURE_TYPE_SI:

            av_log(s->avctx,AV_LOG_DEBUG,"SI\n");

            break;

        case AV_PICTURE_TYPE_SP:

            av_log(s->avctx,AV_LOG_DEBUG,"SP\n");

            break;

        }

        for (y = 0; y < s->mb_height; y++) {

            for (x = 0; x < s->mb_width; x++) {

                if (s->avctx->debug & FF_DEBUG_SKIP) {

                    int count = s->mbskip_table[x + y * s->mb_stride];

                    if (count > 9)

                        count = 9;

                    av_log(s->avctx, AV_LOG_DEBUG, "%1d", count);

                }

                if (s->avctx->debug & FF_DEBUG_QP) {

                    av_log(s->avctx, AV_LOG_DEBUG, "%2d",

                           p->qscale_table[x + y * s->mb_stride]);

                }

                if (s->avctx->debug & FF_DEBUG_MB_TYPE) {

                    int mb_type = p->mb_type[x + y * s->mb_stride];

                    // Type & MV direction

                    if (IS_PCM(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "P");

                    else if (IS_INTRA(mb_type) && IS_ACPRED(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "A");

                    else if (IS_INTRA4x4(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "i");

                    else if (IS_INTRA16x16(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "I");

                    else if (IS_DIRECT(mb_type) && IS_SKIP(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "d");

                    else if (IS_DIRECT(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "D");

                    else if (IS_GMC(mb_type) && IS_SKIP(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "g");

                    else if (IS_GMC(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "G");

                    else if (IS_SKIP(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "S");

                    else if (!USES_LIST(mb_type, 1))

                        av_log(s->avctx, AV_LOG_DEBUG, ">");

                    else if (!USES_LIST(mb_type, 0))

                        av_log(s->avctx, AV_LOG_DEBUG, "<");

                    else {

                        assert(USES_LIST(mb_type, 0) && USES_LIST(mb_type, 1));

                        av_log(s->avctx, AV_LOG_DEBUG, "X");

                    }



                    // segmentation

                    if (IS_8X8(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "+");

                    else if (IS_16X8(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "-");

                    else if (IS_8X16(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "|");

                    else if (IS_INTRA(mb_type) || IS_16X16(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, " ");

                    else

                        av_log(s->avctx, AV_LOG_DEBUG, "?");





                    if (IS_INTERLACED(mb_type))

                        av_log(s->avctx, AV_LOG_DEBUG, "=");

                    else

                        av_log(s->avctx, AV_LOG_DEBUG, " ");

                }

            }

            av_log(s->avctx, AV_LOG_DEBUG, "\n");

        }

    }

}
