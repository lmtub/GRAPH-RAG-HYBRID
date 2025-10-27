int av_vsrc_buffer_add_video_buffer_ref(AVFilterContext *buffer_filter, AVFilterBufferRef *picref)

{

    BufferSourceContext *c = buffer_filter->priv;

    AVFilterLink *outlink = buffer_filter->outputs[0];

    int ret;



    if (c->picref) {

        av_log(buffer_filter, AV_LOG_ERROR,

               "Buffering several frames is not supported. "

               "Please consume all available frames before adding a new one.\n"

            );

        //return -1;

    }



    if (picref->video->w != c->w || picref->video->h != c->h || picref->format != c->pix_fmt) {

        AVFilterContext *scale = buffer_filter->outputs[0]->dst;

        AVFilterLink *link;

        char scale_param[1024];



        av_log(buffer_filter, AV_LOG_INFO,

               "Buffer video input changed from size:%dx%d fmt:%s to size:%dx%d fmt:%s\n",

               c->w, c->h, av_pix_fmt_descriptors[c->pix_fmt].name,

               picref->video->w, picref->video->h, av_pix_fmt_descriptors[picref->format].name);



        if (!scale || strcmp(scale->filter->name, "scale")) {

            AVFilter *f = avfilter_get_by_name("scale");



            av_log(buffer_filter, AV_LOG_INFO, "Inserting scaler filter\n");

            if ((ret = avfilter_open(&scale, f, "Input equalizer")) < 0)

                return ret;



            snprintf(scale_param, sizeof(scale_param)-1, "%d:%d:%s", c->w, c->h, c->sws_param);

            if ((ret = avfilter_init_filter(scale, scale_param, NULL)) < 0) {

                avfilter_free(scale);

                return ret;

            }



            if ((ret = avfilter_insert_filter(buffer_filter->outputs[0], scale, 0, 0)) < 0) {

                avfilter_free(scale);

                return ret;

            }

            scale->outputs[0]->time_base = scale->inputs[0]->time_base;



            scale->outputs[0]->format= c->pix_fmt;

        } else if (!strcmp(scale->filter->name, "scale")) {

            snprintf(scale_param, sizeof(scale_param)-1, "%d:%d:%s",

                     scale->outputs[0]->w, scale->outputs[0]->h, c->sws_param);

            scale->filter->init(scale, scale_param, NULL);

        }



        c->pix_fmt = scale->inputs[0]->format = picref->format;

        c->w       = scale->inputs[0]->w      = picref->video->w;

        c->h       = scale->inputs[0]->h      = picref->video->h;



        link = scale->outputs[0];

        if ((ret =  link->srcpad->config_props(link)) < 0)

            return ret;

    }



    c->picref = avfilter_get_video_buffer(outlink, AV_PERM_WRITE,

                                          picref->video->w, picref->video->h);

    av_image_copy(c->picref->data, c->picref->linesize,

                  picref->data, picref->linesize,

                  picref->format, picref->video->w, picref->video->h);

    avfilter_copy_buffer_ref_props(c->picref, picref);



    return 0;

}
