static int filter_frame(AVFilterLink *link, AVFilterBufferRef *picref)

{

    AVFilterContext *ctx = link->dst;

    YADIFContext *yadif = ctx->priv;



    av_assert0(picref);



    if (picref->video->h < 3 || picref->video->w < 3) {

        av_log(ctx, AV_LOG_ERROR, "Video of less than 3 columns or lines is not supported\n");

        return AVERROR(EINVAL);

    }



    if (yadif->frame_pending)

        return_frame(ctx, 1);



    if (yadif->prev)

        avfilter_unref_buffer(yadif->prev);

    yadif->prev = yadif->cur;

    yadif->cur  = yadif->next;

    yadif->next = picref;



    if (!yadif->cur)

        return 0;



    if (yadif->auto_enable && !yadif->cur->video->interlaced) {

        yadif->out  = avfilter_ref_buffer(yadif->cur, ~AV_PERM_WRITE);

        if (!yadif->out)

            return AVERROR(ENOMEM);



        avfilter_unref_bufferp(&yadif->prev);

        if (yadif->out->pts != AV_NOPTS_VALUE)

            yadif->out->pts *= 2;

        return ff_filter_frame(ctx->outputs[0], yadif->out);

    }



    if (!yadif->prev &&

        !(yadif->prev = avfilter_ref_buffer(yadif->cur, ~AV_PERM_WRITE)))

        return AVERROR(ENOMEM);



    yadif->out = ff_get_video_buffer(ctx->outputs[0], PERM_RWP,

                                     link->w, link->h);

    if (!yadif->out)

        return AVERROR(ENOMEM);



    avfilter_copy_buffer_ref_props(yadif->out, yadif->cur);

    yadif->out->video->interlaced = 0;



    if (yadif->out->pts != AV_NOPTS_VALUE)

        yadif->out->pts *= 2;



    return return_frame(ctx, 0);

}
