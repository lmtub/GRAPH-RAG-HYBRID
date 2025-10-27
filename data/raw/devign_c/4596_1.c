static int teletext_decode_frame(AVCodecContext *avctx, void *data, int *data_size, AVPacket *pkt)

{

    TeletextContext *ctx = avctx->priv_data;

    AVSubtitle      *sub = data;

    const uint8_t   *buf = pkt->data;

    int             left = pkt->size;

    uint8_t         pesheader[45] = {0x00, 0x00, 0x01, 0xbd, 0x00, 0x00, 0x85, 0x80, 0x24, 0x21, 0x00, 0x01, 0x00, 0x01};

    int             pesheader_size = sizeof(pesheader);

    const uint8_t   *pesheader_buf = pesheader;

    int             ret = 0;



    if (!ctx->vbi) {

        if (!(ctx->vbi = vbi_decoder_new()))

            return AVERROR(ENOMEM);

        if (!vbi_event_handler_add(ctx->vbi, VBI_EVENT_TTX_PAGE, handler, ctx)) {

            vbi_decoder_delete(ctx->vbi);

            ctx->vbi = NULL;

            return AVERROR(ENOMEM);

        }

    }

    if (!ctx->dx && (!(ctx->dx = vbi_dvb_pes_demux_new (/* callback */ NULL, NULL))))

        return AVERROR(ENOMEM);



    if (avctx->pkt_timebase.den && pkt->pts != AV_NOPTS_VALUE)

        ctx->pts = av_rescale_q(pkt->pts, avctx->pkt_timebase, AV_TIME_BASE_Q);



    if (left) {

        // We allow unreasonably big packets, even if the standard only allows a max size of 1472

        if ((pesheader_size + left) < 184 || (pesheader_size + left) > 65504 || (pesheader_size + left) % 184 != 0)

            return AVERROR_INVALIDDATA;



        memset(pesheader + 14, 0xff, pesheader_size - 14);

        AV_WB16(pesheader + 4, left + pesheader_size - 6);



        /* PTS is deliberately left as 0 in the PES header, otherwise libzvbi uses

         * it to detect dropped frames. Unforunatey the guessed packet PTS values

         * (see mpegts demuxer) are not accurate enough to pass that test. */

        vbi_dvb_demux_cor(ctx->dx, ctx->sliced, 64, NULL, &pesheader_buf, &pesheader_size);



        ctx->handler_ret = pkt->size;



        while (left > 0) {

            int64_t pts = 0;

            unsigned int lines = vbi_dvb_demux_cor(ctx->dx, ctx->sliced, 64, &pts, &buf, &left);

            av_dlog(avctx, "ctx=%p buf_size=%d left=%u lines=%u pts=%f pkt_pts=%f\n",

                    ctx, pkt->size, left, lines, (double)pts/90000.0, (double)pkt->pts/90000.0);

            if (lines > 0) {

#ifdef DEBUGx

                int i;

                for(i=0; i<lines; ++i)

                    av_log(avctx, AV_LOG_DEBUG,

                           "lines=%d id=%x\n", i, ctx->sliced[i].id);

#endif

                vbi_decode(ctx->vbi, ctx->sliced, lines, (double)pts/90000.0);

                ctx->lines_processed += lines;

            }

        }

        ctx->pts = AV_NOPTS_VALUE;

        ret = ctx->handler_ret;

    }



    if (ret < 0)

        return ret;



    // is there a subtitle to pass?

    if (ctx->nb_pages) {

        int i;

        sub->format = ctx->format_id;

        sub->start_display_time = 0;

        sub->end_display_time = ctx->sub_duration;

        sub->num_rects = 0;

        sub->pts = ctx->pages->pts;



        if (ctx->pages->sub_rect->type != SUBTITLE_NONE) {

            sub->rects = av_malloc(sizeof(*sub->rects) * 1);

            if (sub->rects) {

                sub->num_rects = 1;

                sub->rects[0] = ctx->pages->sub_rect;

            } else {

                ret = AVERROR(ENOMEM);

            }

        } else {

            av_log(avctx, AV_LOG_DEBUG, "sending empty sub\n");

            sub->rects = NULL;

        }

        if (!sub->rects) // no rect was passed

            subtitle_rect_free(&ctx->pages->sub_rect);



        for (i = 0; i < ctx->nb_pages - 1; i++)

            ctx->pages[i] = ctx->pages[i + 1];

        ctx->nb_pages--;



        if (ret >= 0)

            *data_size = 1;

    } else

        *data_size = 0;



    return ret;

}
