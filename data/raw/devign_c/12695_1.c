int avfilter_graph_queue_command(AVFilterGraph *graph, const char *target, const char *command, const char *arg, int flags, double ts)
{
    int i;
    if(!graph)
        return 0;
    for (i = 0; i < graph->nb_filters; i++) {
        AVFilterContext *filter = graph->filters[i];
        if(filter && (!strcmp(target, "all") || !strcmp(target, filter->name) || !strcmp(target, filter->filter->name))){
            AVFilterCommand **queue = &filter->command_queue, *next;
            while (*queue && (*queue)->time <= ts)
                queue = &(*queue)->next;
            next = *queue;
            *queue = av_mallocz(sizeof(AVFilterCommand));
            (*queue)->command = av_strdup(command);
            (*queue)->arg     = av_strdup(arg);
            (*queue)->time    = ts;
            (*queue)->flags   = flags;
            (*queue)->next    = next;
            if(flags & AVFILTER_CMD_FLAG_ONE)
                return 0;
        }
    }
    return 0;
}