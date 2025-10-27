static int decode(AVCodecContext *avctx, void *data, int *got_sub, AVPacket *avpkt)

{

    CCaptionSubContext *ctx = avctx->priv_data;

    AVSubtitle *sub = data;

    const int64_t start_time = sub->pts;

    uint8_t *bptr = NULL;

    int len = avpkt->size;

    int ret = 0;

    int i;



    if (ctx->pktbuf->size < len) {

        ret = av_buffer_realloc(&ctx->pktbuf, len);

         if (ret < 0) {

            av_log(ctx, AV_LOG_WARNING, "Insufficient Memory of %d truncated to %d\n", len, ctx->pktbuf->size);

            len = ctx->pktbuf->size;

            ret = 0;

        }

    }

    memcpy(ctx->pktbuf->data, avpkt->data, len);

    bptr = ctx->pktbuf->data;



    for (i  = 0; i < len; i += 3) {

        uint8_t cc_type = *(bptr + i) & 3;

        if (validate_cc_data_pair(bptr + i))

            continue;

        /* ignoring data field 1 */

        if(cc_type == 1)

            continue;

        else

            process_cc608(ctx, start_time, *(bptr + i + 1) & 0x7f, *(bptr + i + 2) & 0x7f);



        if (!ctx->buffer_changed)

            continue;

        ctx->buffer_changed = 0;



        if (*ctx->buffer.str || ctx->real_time)

        {

            ff_dlog(ctx, "cdp writing data (%s)\n",ctx->buffer.str);

            ret = ff_ass_add_rect(sub, ctx->buffer.str, ctx->readorder++, 0, NULL, NULL);

            if (ret < 0)

                return ret;

            sub->pts = ctx->start_time;

            if (!ctx->real_time)

                sub->end_display_time = av_rescale_q(ctx->end_time - ctx->start_time,

                                                     AV_TIME_BASE_Q, ms_tb);

            else

                sub->end_display_time = -1;

            ctx->buffer_changed = 0;

            ctx->last_real_time = sub->pts;

            ctx->screen_touched = 0;

        }

    }



    if (ctx->real_time && ctx->screen_touched &&

        sub->pts > ctx->last_real_time + av_rescale_q(200, ms_tb, AV_TIME_BASE_Q)) {

        ctx->last_real_time = sub->pts;

        ctx->screen_touched = 0;



        capture_screen(ctx);

        ctx->buffer_changed = 0;



        ret = ff_ass_add_rect(sub, ctx->buffer.str, ctx->readorder++, 0, NULL, NULL);

        if (ret < 0)

            return ret;

        sub->end_display_time = -1;

    }



    *got_sub = sub->num_rects > 0;

    return ret;

}
