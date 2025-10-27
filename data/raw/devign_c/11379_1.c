int avfilter_graph_add_filter(AVFilterGraph *graph, AVFilterContext *filter)

{

    graph->filters = av_realloc(graph->filters,

                                sizeof(AVFilterContext*) * ++graph->filter_count);



    if (!graph->filters)

        return AVERROR(ENOMEM);



    graph->filters[graph->filter_count - 1] = filter;



    return 0;

}
