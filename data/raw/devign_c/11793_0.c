static int filter_samples(AVFilterLink *inlink, AVFilterBufferRef *insamplesref)

{

    AResampleContext *aresample = inlink->dst->priv;

    const int n_in  = insamplesref->audio->nb_samples;

    int n_out       = FFMAX(n_in * aresample->ratio * 2, 1);

    AVFilterLink *const outlink = inlink->dst->outputs[0];

    AVFilterBufferRef *outsamplesref = ff_get_audio_buffer(outlink, AV_PERM_WRITE, n_out);

    int ret;



    if(!outsamplesref)

        return AVERROR(ENOMEM);



    avfilter_copy_buffer_ref_props(outsamplesref, insamplesref);

    outsamplesref->format                = outlink->format;

    outsamplesref->audio->channel_layout = outlink->channel_layout;

    outsamplesref->audio->sample_rate    = outlink->sample_rate;



    if(insamplesref->pts != AV_NOPTS_VALUE) {

        int64_t inpts = av_rescale(insamplesref->pts, inlink->time_base.num * (int64_t)outlink->sample_rate * inlink->sample_rate, inlink->time_base.den);

        int64_t outpts= swr_next_pts(aresample->swr, inpts);

        aresample->next_pts =

        outsamplesref->pts  = (outpts + inlink->sample_rate/2) / inlink->sample_rate;

    } else {

        outsamplesref->pts  = AV_NOPTS_VALUE;

    }



    n_out = swr_convert(aresample->swr, outsamplesref->extended_data, n_out,

                                 (void *)insamplesref->extended_data, n_in);

    if (n_out <= 0) {

        avfilter_unref_buffer(outsamplesref);

        avfilter_unref_buffer(insamplesref);

        return 0;

    }



    outsamplesref->audio->nb_samples  = n_out;



    ret = ff_filter_samples(outlink, outsamplesref);

    aresample->req_fullfilled= 1;

    avfilter_unref_buffer(insamplesref);

    return ret;

}
