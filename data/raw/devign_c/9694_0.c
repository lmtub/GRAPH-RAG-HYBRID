static int default_start_frame(AVFilterLink *inlink, AVFilterBufferRef *picref)

{

    AVFilterLink *outlink = NULL;



    if (inlink->dst->nb_outputs)

        outlink = inlink->dst->outputs[0];



    if (outlink) {

        outlink->out_buf = ff_get_video_buffer(outlink, AV_PERM_WRITE, outlink->w, outlink->h);

        if (!outlink->out_buf)

            return AVERROR(ENOMEM);



        avfilter_copy_buffer_ref_props(outlink->out_buf, picref);

        return ff_start_frame(outlink, avfilter_ref_buffer(outlink->out_buf, ~0));

    }

    return 0;

}
