int avfilter_copy_frame_props(AVFilterBufferRef *dst, const AVFrame *src)

{

    dst->pts    = src->pts;

    dst->pos    = av_frame_get_pkt_pos(src);

    dst->format = src->format;



    switch (dst->type) {

    case AVMEDIA_TYPE_VIDEO:

        dst->video->w                   = src->width;

        dst->video->h                   = src->height;

        dst->video->sample_aspect_ratio = src->sample_aspect_ratio;

        dst->video->interlaced          = src->interlaced_frame;

        dst->video->top_field_first     = src->top_field_first;

        dst->video->key_frame           = src->key_frame;

        dst->video->pict_type           = src->pict_type;

        av_freep(&dst->video->qp_table);

        dst->video->qp_table_linesize = 0;

        if (src->qscale_table) {

            int qsize = src->qstride ? src->qstride * ((src->height+15)/16) : (src->width+15)/16;

            dst->video->qp_table        = av_malloc(qsize);

            if(!dst->video->qp_table)

                return AVERROR(ENOMEM);

            dst->video->qp_table_linesize       = src->qstride;


            memcpy(dst->video->qp_table, src->qscale_table, qsize);

        }

        break;

    case AVMEDIA_TYPE_AUDIO:

        dst->audio->sample_rate         = src->sample_rate;

        dst->audio->channel_layout      = src->channel_layout;

        break;

    default:

        return AVERROR(EINVAL);

    }



    return 0;

}