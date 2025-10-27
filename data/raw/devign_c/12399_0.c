int avfilter_parse_graph(AVFilterGraph *graph, const char *filters,

                         AVFilterInOut *openLinks, AVClass *log_ctx)

{

    int index = 0;

    char chr = 0;

    int pad = 0;



    AVFilterInOut *currInputs = NULL;



    do {

        AVFilterContext *filter;

        filters += consume_whitespace(filters);



        pad = parse_inputs(&filters, &currInputs, &openLinks, log_ctx);



        if(pad < 0)

            goto fail;



        if(!(filter = parse_filter(&filters, graph, index, log_ctx)))

            goto fail;



        if(filter->input_count == 1 && !currInputs && !index) {

            // First input can be ommitted if it is "[in]"

            const char *tmp = "[in]";

            pad = parse_inputs(&tmp, &currInputs, &openLinks, log_ctx);

            if(pad < 0)

                goto fail;

        }



        if(link_filter_inouts(filter, &currInputs, &openLinks, log_ctx) < 0)

            goto fail;



        pad = parse_outputs(&filters, &currInputs, &openLinks, log_ctx);



        if(pad < 0)

            goto fail;



        filters += consume_whitespace(filters);

        chr = *filters++;



        if(chr == ';' && currInputs) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Could not find a output to link when parsing \"%s\"\n",

                   filters - 1);

            goto fail;

        }

        index++;

    } while(chr == ',' || chr == ';');



    if(openLinks && !strcmp(openLinks->name, "out") && currInputs) {

        // Last output can be ommitted if it is "[out]"

        const char *tmp = "[out]";

        if(parse_outputs(&tmp, &currInputs, &openLinks, log_ctx) < 0)

            goto fail;

    }



    return 0;



 fail:

    avfilter_destroy_graph(graph);

    free_inout(openLinks);

    free_inout(currInputs);

    return -1;

}
