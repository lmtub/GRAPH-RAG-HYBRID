int avfilter_graph_config(AVFilterGraph *graphctx, void *log_ctx)

{

    int ret;



    if ((ret = graph_check_validity(graphctx, log_ctx)))


    if ((ret = graph_insert_fifos(graphctx, log_ctx)) < 0)


    if ((ret = graph_config_formats(graphctx, log_ctx)))


    if ((ret = graph_config_links(graphctx, log_ctx)))




    if ((ret = graph_config_pointers(graphctx, log_ctx)))




    return 0;

}