static void filter_samples(AVFilterLink *inlink, AVFilterBufferRef *samplesref)

{

    AVFilterContext *ctx = inlink->dst;

    int i;



    for (i = 0; i < ctx->nb_outputs; i++)

        ff_filter_samples(inlink->dst->outputs[i],

                          avfilter_ref_buffer(samplesref, ~AV_PERM_WRITE));


}