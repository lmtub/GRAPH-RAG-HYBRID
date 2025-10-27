int avfilter_graph_parse(AVFilterGraph *graph, const char *filters,

                         AVFilterInOut **open_inputs_ptr, AVFilterInOut **open_outputs_ptr,

                         void *log_ctx)

{

    int index = 0, ret = 0;

    char chr = 0;



    AVFilterInOut *curr_inputs = NULL;

    AVFilterInOut *open_inputs  = open_inputs_ptr  ? *open_inputs_ptr  : NULL;

    AVFilterInOut *open_outputs = open_outputs_ptr ? *open_outputs_ptr : NULL;



    do {

        AVFilterContext *filter;

        const char *filterchain = filters;

        filters += strspn(filters, WHITESPACES);



        if ((ret = parse_inputs(&filters, &curr_inputs, &open_outputs, log_ctx)) < 0)

            goto end;



        if ((ret = parse_filter(&filter, &filters, graph, index, log_ctx)) < 0)

            goto end;



        if (filter->input_count == 1 && !curr_inputs && !index) {

            /* First input pad, assume it is "[in]" if not specified */

            const char *tmp = "[in]";

            if ((ret = parse_inputs(&tmp, &curr_inputs, &open_outputs, log_ctx)) < 0)

                goto end;

        }



        if ((ret = link_filter_inouts(filter, &curr_inputs, &open_inputs, log_ctx)) < 0)

            goto end;



        if ((ret = parse_outputs(&filters, &curr_inputs, &open_inputs, &open_outputs,

                                 log_ctx)) < 0)

            goto end;



        filters += strspn(filters, WHITESPACES);

        chr = *filters++;



        if (chr == ';' && curr_inputs) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Invalid filterchain containing an unlabelled output pad: \"%s\"\n",

                   filterchain);

            ret = AVERROR(EINVAL);

            goto end;

        }

        index++;

    } while (chr == ',' || chr == ';');



    if (chr) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Unable to parse graph description substring: \"%s\"\n",

               filters - 1);

        ret = AVERROR(EINVAL);

        goto end;

    }



    if (open_inputs && !strcmp(open_inputs->name, "out") && curr_inputs) {

        /* Last output pad, assume it is "[out]" if not specified */

        const char *tmp = "[out]";

        if ((ret = parse_outputs(&tmp, &curr_inputs, &open_inputs, &open_outputs,

                                 log_ctx)) < 0)

            goto end;

    }



end:

    /* clear open_in/outputs only if not passed as parameters */

    if (open_inputs_ptr) *open_inputs_ptr = open_inputs;

    else avfilter_inout_free(&open_inputs);

    if (open_outputs_ptr) *open_outputs_ptr = open_outputs;

    else avfilter_inout_free(&open_outputs);

    avfilter_inout_free(&curr_inputs);



    if (ret < 0) {

        for (; graph->filter_count > 0; graph->filter_count--)

            avfilter_free(graph->filters[graph->filter_count - 1]);

        av_freep(&graph->filters);

    }

    return ret;

}
