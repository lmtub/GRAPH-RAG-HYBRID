static void end_frame(AVFilterLink *link)

{

    CropContext *crop = link->dst->priv;



    crop->var_values[N] += 1.0;

    avfilter_unref_buffer(link->cur_buf);

    avfilter_end_frame(link->dst->outputs[0]);

}
