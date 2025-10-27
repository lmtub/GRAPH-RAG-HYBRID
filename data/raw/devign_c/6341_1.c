void avfilter_destroy(AVFilterContext *filter)

{

    int i;



    if(filter->filter->uninit)

        filter->filter->uninit(filter);



    for(i = 0; i < filter->input_count; i ++) {

        if(filter->inputs[i]) {


            filter->inputs[i]->src->outputs[filter->inputs[i]->srcpad] = NULL;

            avfilter_formats_unref(&filter->inputs[i]->in_formats);

            avfilter_formats_unref(&filter->inputs[i]->out_formats);

        }

        av_freep(&filter->inputs[i]);

    }

    for(i = 0; i < filter->output_count; i ++) {

        if(filter->outputs[i]) {

            if (filter->outputs[i]->dst)

            filter->outputs[i]->dst->inputs[filter->outputs[i]->dstpad] = NULL;

            avfilter_formats_unref(&filter->outputs[i]->in_formats);

            avfilter_formats_unref(&filter->outputs[i]->out_formats);

        }

        av_freep(&filter->outputs[i]);

    }



    av_freep(&filter->name);

    av_freep(&filter->input_pads);

    av_freep(&filter->output_pads);

    av_freep(&filter->inputs);

    av_freep(&filter->outputs);

    av_freep(&filter->priv);

    av_free(filter);

}