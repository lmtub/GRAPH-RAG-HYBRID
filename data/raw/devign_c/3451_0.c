static int ff_filter_frame_framed(AVFilterLink *link, AVFrame *frame)

{

    int (*filter_frame)(AVFilterLink *, AVFrame *);

    AVFilterContext *dstctx = link->dst;

    AVFilterPad *dst = link->dstpad;

    AVFrame *out = NULL;

    int ret;

    AVFilterCommand *cmd= link->dst->command_queue;

    int64_t pts;



    if (link->closed) {

        av_frame_free(&frame);

        return AVERROR_EOF;

    }



    if (!(filter_frame = dst->filter_frame))

        filter_frame = default_filter_frame;



    /* copy the frame if needed */

    if (dst->needs_writable && !av_frame_is_writable(frame)) {

        av_log(link->dst, AV_LOG_DEBUG, "Copying data in avfilter.\n");



        switch (link->type) {

        case AVMEDIA_TYPE_VIDEO:

            out = ff_get_video_buffer(link, link->w, link->h);

            break;

        case AVMEDIA_TYPE_AUDIO:

            out = ff_get_audio_buffer(link, frame->nb_samples);

            break;

        default:

            ret = AVERROR(EINVAL);

            goto fail;

        }

        if (!out) {

            ret = AVERROR(ENOMEM);

            goto fail;

        }



        ret = av_frame_copy_props(out, frame);

        if (ret < 0)

            goto fail;



        switch (link->type) {

        case AVMEDIA_TYPE_VIDEO:

            av_image_copy(out->data, out->linesize, (const uint8_t **)frame->data, frame->linesize,

                          frame->format, frame->width, frame->height);

            break;

        case AVMEDIA_TYPE_AUDIO:

            av_samples_copy(out->extended_data, frame->extended_data,

                            0, 0, frame->nb_samples,

                            av_get_channel_layout_nb_channels(frame->channel_layout),

                            frame->format);

            break;

        default:

            ret = AVERROR(EINVAL);

            goto fail;

        }



        av_frame_free(&frame);

    } else

        out = frame;



    while(cmd && cmd->time <= out->pts * av_q2d(link->time_base)){

        av_log(link->dst, AV_LOG_DEBUG,

               "Processing command time:%f command:%s arg:%s\n",

               cmd->time, cmd->command, cmd->arg);

        avfilter_process_command(link->dst, cmd->command, cmd->arg, 0, 0, cmd->flags);

        ff_command_queue_pop(link->dst);

        cmd= link->dst->command_queue;

    }



    pts = out->pts;

    if (dstctx->enable_str) {

        int64_t pos = av_frame_get_pkt_pos(out);

        dstctx->var_values[VAR_N] = link->frame_count;

        dstctx->var_values[VAR_T] = pts == AV_NOPTS_VALUE ? NAN : pts * av_q2d(link->time_base);

        dstctx->var_values[VAR_W] = link->w;

        dstctx->var_values[VAR_H] = link->h;

        dstctx->var_values[VAR_POS] = pos == -1 ? NAN : pos;



        dstctx->is_disabled = fabs(av_expr_eval(dstctx->enable, dstctx->var_values, NULL)) < 0.5;

        if (dstctx->is_disabled &&

            (dstctx->filter->flags & AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC))

            filter_frame = default_filter_frame;

    }

    ret = filter_frame(link, out);

    link->frame_count++;

    link->frame_requested = 0;

    ff_update_link_current_pts(link, pts);

    return ret;



fail:

    av_frame_free(&out);

    av_frame_free(&frame);

    return ret;

}
