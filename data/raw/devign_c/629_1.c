static int ff_filter_frame_framed(AVFilterLink *link, AVFilterBufferRef *frame)

{

    int (*filter_frame)(AVFilterLink *, AVFilterBufferRef *);

    AVFilterPad *src = link->srcpad;

    AVFilterPad *dst = link->dstpad;

    AVFilterBufferRef *out;

    int perms, ret;

    AVFilterCommand *cmd= link->dst->command_queue;

    int64_t pts;



    if (link->closed) {

        avfilter_unref_buffer(frame);

        return AVERROR_EOF;

    }



    if (!(filter_frame = dst->filter_frame))

        filter_frame = default_filter_frame;



    av_assert1((frame->perms & src->min_perms) == src->min_perms);

    frame->perms &= ~ src->rej_perms;

    perms = frame->perms;



    if (frame->linesize[0] < 0)

        perms |= AV_PERM_NEG_LINESIZES;



    /* prepare to copy the frame if the buffer has insufficient permissions */

    if ((dst->min_perms & perms) != dst->min_perms ||

        dst->rej_perms & perms) {

        av_log(link->dst, AV_LOG_DEBUG,

               "Copying data in avfilter (have perms %x, need %x, reject %x)\n",

               perms, link->dstpad->min_perms, link->dstpad->rej_perms);



        /* Maybe use ff_copy_buffer_ref instead? */

        switch (link->type) {

        case AVMEDIA_TYPE_VIDEO:

            out = ff_get_video_buffer(link, dst->min_perms,

                                      link->w, link->h);

            break;

        case AVMEDIA_TYPE_AUDIO:

            out = ff_get_audio_buffer(link, dst->min_perms,

                                      frame->audio->nb_samples);

            break;

        default: return AVERROR(EINVAL);

        }

        if (!out) {

            avfilter_unref_buffer(frame);

            return AVERROR(ENOMEM);

        }

        avfilter_copy_buffer_ref_props(out, frame);



        switch (link->type) {

        case AVMEDIA_TYPE_VIDEO:

            av_image_copy(out->data, out->linesize, frame->data, frame->linesize,

                          frame->format, frame->video->w, frame->video->h);

            break;

        case AVMEDIA_TYPE_AUDIO:

            av_samples_copy(out->extended_data, frame->extended_data,

                            0, 0, frame->audio->nb_samples,

                            av_get_channel_layout_nb_channels(frame->audio->channel_layout),

                            frame->format);

            break;

        default: return AVERROR(EINVAL);

        }



        avfilter_unref_buffer(frame);

    } else

        out = frame;



    while(cmd && cmd->time <= frame->pts * av_q2d(link->time_base)){

        av_log(link->dst, AV_LOG_DEBUG,

               "Processing command time:%f command:%s arg:%s\n",

               cmd->time, cmd->command, cmd->arg);

        avfilter_process_command(link->dst, cmd->command, cmd->arg, 0, 0, cmd->flags);

        ff_command_queue_pop(link->dst);

        cmd= link->dst->command_queue;

    }



    pts = out->pts;

    ret = filter_frame(link, out);

    ff_update_link_current_pts(link, pts);

    return ret;

}
