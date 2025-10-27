static void default_end_frame(AVFilterLink *inlink)

{

    AVFilterLink *outlink = NULL;



    if (inlink->dst->nb_outputs)

        outlink = inlink->dst->outputs[0];



    if (outlink) {

        if (outlink->out_buf) {

            avfilter_unref_buffer(outlink->out_buf);

            outlink->out_buf = NULL;

        }

        ff_end_frame(outlink);

    }

}
