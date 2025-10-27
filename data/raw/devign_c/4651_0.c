int ff_avfilter_graph_config_formats(AVFilterGraph *graph, AVClass *log_ctx)

{

    int ret;



    /* find supported formats from sub-filters, and merge along links */

    if ((ret = query_formats(graph, log_ctx)) < 0)

        return ret;



    /* Once everything is merged, it's possible that we'll still have

     * multiple valid media format choices. We pick the first one. */

    pick_formats(graph);



    return 0;

}
