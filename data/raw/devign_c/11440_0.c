int avfilter_config_links(AVFilterContext *filter)

{

    int (*config_link)(AVFilterLink *);

    unsigned i;

    int ret;



    for (i = 0; i < filter->nb_inputs; i ++) {

        AVFilterLink *link = filter->inputs[i];

        AVFilterLink *inlink = link->src->nb_inputs ?

            link->src->inputs[0] : NULL;



        if (!link) continue;



        link->current_pts = AV_NOPTS_VALUE;



        switch (link->init_state) {

        case AVLINK_INIT:

            continue;

        case AVLINK_STARTINIT:

            av_log(filter, AV_LOG_INFO, "circular filter chain detected\n");

            return 0;

        case AVLINK_UNINIT:

            link->init_state = AVLINK_STARTINIT;



            if ((ret = avfilter_config_links(link->src)) < 0)

                return ret;



            if (!(config_link = link->srcpad->config_props)) {

                if (link->src->nb_inputs != 1) {

                    av_log(link->src, AV_LOG_ERROR, "Source filters and filters "

                                                    "with more than one input "

                                                    "must set config_props() "

                                                    "callbacks on all outputs\n");

                    return AVERROR(EINVAL);

                }

            } else if ((ret = config_link(link)) < 0) {

                av_log(link->src, AV_LOG_ERROR,

                       "Failed to configure output pad on %s\n",

                       link->src->name);

                return ret;

            }



            switch (link->type) {

            case AVMEDIA_TYPE_VIDEO:

                if (!link->time_base.num && !link->time_base.den)

                    link->time_base = inlink ? inlink->time_base : AV_TIME_BASE_Q;



                if (!link->sample_aspect_ratio.num && !link->sample_aspect_ratio.den)

                    link->sample_aspect_ratio = inlink ?

                        inlink->sample_aspect_ratio : (AVRational){1,1};



                if (inlink && !link->frame_rate.num && !link->frame_rate.den)

                    link->frame_rate = inlink->frame_rate;



                if (inlink) {

                    if (!link->w)

                        link->w = inlink->w;

                    if (!link->h)

                        link->h = inlink->h;

                } else if (!link->w || !link->h) {

                    av_log(link->src, AV_LOG_ERROR,

                           "Video source filters must set their output link's "

                           "width and height\n");

                    return AVERROR(EINVAL);

                }

                break;



            case AVMEDIA_TYPE_AUDIO:

                if (inlink) {

                    if (!link->sample_rate)

                        link->sample_rate = inlink->sample_rate;

                    if (!link->time_base.num && !link->time_base.den)

                        link->time_base = inlink->time_base;

                    if (!link->channel_layout)

                        link->channel_layout = inlink->channel_layout;

                } else if (!link->sample_rate) {

                    av_log(link->src, AV_LOG_ERROR,

                           "Audio source filters must set their output link's "

                           "sample_rate\n");

                    return AVERROR(EINVAL);

                }



                if (!link->time_base.num && !link->time_base.den)

                    link->time_base = (AVRational) {1, link->sample_rate};

            }



            if ((config_link = link->dstpad->config_props))

                if ((ret = config_link(link)) < 0) {

                    av_log(link->src, AV_LOG_ERROR,

                           "Failed to configure input pad on %s\n",

                           link->dst->name);

                    return ret;

                }



            link->init_state = AVLINK_INIT;

        }

    }



    return 0;

}
