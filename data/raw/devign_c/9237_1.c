static AVFilterContext *create_filter(AVFilterGraph *ctx, int index,

                                      const char *name, const char *args,

                                      AVClass *log_ctx)

{

    AVFilterContext *filt;



    AVFilter *filterdef;

    char inst_name[30];



    snprintf(inst_name, sizeof(inst_name), "Parsed filter %d", index);



    filterdef = avfilter_get_by_name(name);



    if(!filterdef) {

        av_log(log_ctx, AV_LOG_ERROR,

               "no such filter: '%s'\n", name);

        return NULL;

    }



    filt = avfilter_open(filterdef, inst_name);

    if(!filt) {

        av_log(log_ctx, AV_LOG_ERROR,

               "error creating filter '%s'\n", name);

        return NULL;

    }



    if(avfilter_graph_add_filter(ctx, filt) < 0)

        return NULL;



    if(avfilter_init_filter(filt, args, NULL)) {

        av_log(log_ctx, AV_LOG_ERROR,

               "error initializing filter '%s' with args '%s'\n", name, args);

        return NULL;

    }



    return filt;

}
