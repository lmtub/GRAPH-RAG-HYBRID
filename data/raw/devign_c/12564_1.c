int avfilter_graph_send_command(AVFilterGraph *graph, const char *target, const char *cmd, const char *arg, char *res, int res_len, int flags)

{

    int i, r = AVERROR(ENOSYS);



    if(!graph)

        return r;



    if((flags & AVFILTER_CMD_FLAG_ONE) && !(flags & AVFILTER_CMD_FLAG_FAST)) {

        r=avfilter_graph_send_command(graph, target, cmd, arg, res, res_len, flags | AVFILTER_CMD_FLAG_FAST);

        if(r != AVERROR(ENOSYS))

            return r;

    }



    if(res_len && res)

        res[0]= 0;



    for (i = 0; i < graph->filter_count; i++) {

        AVFilterContext *filter = graph->filters[i];

        if(!strcmp(target, "all") || !strcmp(target, filter->name) || !strcmp(target, filter->filter->name)){

            r = avfilter_process_command(filter, cmd, arg, res, res_len, flags);

            if(r != AVERROR(ENOSYS)) {

                if((flags & AVFILTER_CMD_FLAG_ONE) || r<0)

                    return r;

            }

        }

    }



    return r;

}
