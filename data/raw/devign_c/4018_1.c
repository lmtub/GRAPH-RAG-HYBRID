static int filter_samples(AVFilterLink *inlink, AVFilterBufferRef *insamples)

{

    AVFilterContext *ctx = inlink->dst;

    AVFilterLink *outlink = ctx->outputs[0];

    ShowWavesContext *showwaves = ctx->priv;

    const int nb_samples = insamples->audio->nb_samples;

    AVFilterBufferRef *outpicref = showwaves->outpicref;

    int linesize = outpicref ? outpicref->linesize[0] : 0;

    int16_t *p = (int16_t *)insamples->data[0];

    int nb_channels = av_get_channel_layout_nb_channels(insamples->audio->channel_layout);

    int i, j, h;

    const int n = showwaves->n;

    const int x = 255 / (nb_channels * n); /* multiplication factor, pre-computed to avoid in-loop divisions */



    /* draw data in the buffer */

    for (i = 0; i < nb_samples; i++) {

        if (showwaves->buf_idx == 0 && showwaves->sample_count_mod == 0) {

            showwaves->outpicref = outpicref =

                ff_get_video_buffer(outlink, AV_PERM_WRITE|AV_PERM_ALIGN,

                                    outlink->w, outlink->h);

            outpicref->video->w = outlink->w;

            outpicref->video->h = outlink->h;

            outpicref->pts = insamples->pts +

                             av_rescale_q((p - (int16_t *)insamples->data[0]) / nb_channels,

                                          (AVRational){ 1, inlink->sample_rate },

                                          outlink->time_base);

            outlink->out_buf = outpicref;

            linesize = outpicref->linesize[0];

            memset(outpicref->data[0], 0, showwaves->h*linesize);

        }

        for (j = 0; j < nb_channels; j++) {

            h = showwaves->h/2 - av_rescale(*p++, showwaves->h/2, MAX_INT16);

            if (h >= 0 && h < outlink->h)

                *(outpicref->data[0] + showwaves->buf_idx + h * linesize) += x;

        }

        showwaves->sample_count_mod++;

        if (showwaves->sample_count_mod == n) {

            showwaves->sample_count_mod = 0;

            showwaves->buf_idx++;

        }

        if (showwaves->buf_idx == showwaves->w)

            push_frame(outlink);

    }



    avfilter_unref_buffer(insamples);

    return 0;

}
