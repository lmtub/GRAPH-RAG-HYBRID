static int filter_frame(AVFilterLink *inlink, AVFrame *inpicref)

{

    AVFilterContext *ctx = inlink->dst;

    SeparateFieldsContext *sf = ctx->priv;

    AVFilterLink *outlink = ctx->outputs[0];

    AVFrame *second;

    int i, ret;



    inpicref->height = outlink->h;

    inpicref->interlaced_frame = 0;



    second = av_frame_clone(inpicref);

    if (!second)

        return AVERROR(ENOMEM);



    for (i = 0; i < sf->nb_planes; i++) {

        if (!inpicref->top_field_first)

            inpicref->data[i] = inpicref->data[i] + inpicref->linesize[i];

        else

            second->data[i] = second->data[i] + second->linesize[i];

        inpicref->linesize[i] *= 2;

        second->linesize[i]   *= 2;

    }



    inpicref->pts = outlink->frame_count * sf->ts_unit;

    ret = ff_filter_frame(outlink, inpicref);

    if (ret < 0)

        return ret;



    second->pts = outlink->frame_count * sf->ts_unit;

    return ff_filter_frame(outlink, second);

}
