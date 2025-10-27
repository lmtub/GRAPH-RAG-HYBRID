int avfilter_parse_graph(AVFilterGraph *graph, const char *filters,

                         AVFilterInOut *open_inputs,

                         AVFilterInOut *open_outputs, AVClass *log_ctx)

{

    int index = 0;

    char chr = 0;



    AVFilterInOut *curr_inputs = NULL;



    do {

        AVFilterContext *filter;

        filters += consume_whitespace(filters);



        if(parse_inputs(&filters, &curr_inputs, &open_outputs, log_ctx) < 0)

            goto fail;



        filter = parse_filter(&filters, graph, index, log_ctx);



        if(!filter)

            goto fail;



        if(filter->input_count == 1 && !curr_inputs && !index) {

            /* First input can be omitted if it is "[in]" */

            const char *tmp = "[in]";

            if(parse_inputs(&tmp, &curr_inputs, &open_outputs, log_ctx) < 0)

                goto fail;

        }



        if(link_filter_inouts(filter, &curr_inputs, &open_inputs, log_ctx) < 0)

            goto fail;



        if(parse_outputs(&filters, &curr_inputs, &open_inputs, &open_outputs,

                         log_ctx) < 0)

            goto fail;



        filters += consume_whitespace(filters);

        chr = *filters++;



        if(chr == ';' && curr_inputs) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Could not find a output to link when parsing \"%s\"\n",

                   filters - 1);

            goto fail;

        }

        index++;

    } while(chr == ',' || chr == ';');



    if (*filters) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Unable to parse graph description substring: \"%s\"\n",

               filters - 1);

        goto fail;

    }



    if(open_inputs && !strcmp(open_inputs->name, "out") && curr_inputs) {

        /* Last output can be omitted if it is "[out]" */

        const char *tmp = "[out]";

        if(parse_outputs(&tmp, &curr_inputs, &open_inputs,

                         &open_outputs, log_ctx) < 0)

            goto fail;

    }



    return 0;



 fail:

    avfilter_destroy_graph(graph);

    free_inout(open_inputs);

    free_inout(open_outputs);

    free_inout(curr_inputs);

    return -1;

}
