static int create_filter(AVFilterContext **filt_ctx, AVFilterGraph *ctx, int index,

                         const char *filt_name, const char *args, AVClass *log_ctx)

{

    AVFilter *filt;

    char inst_name[30];

    char tmp_args[256];

    int ret;



    snprintf(inst_name, sizeof(inst_name), "Parsed filter %d %s", index, filt_name);



    filt = avfilter_get_by_name(filt_name);



    if (!filt) {

        av_log(log_ctx, AV_LOG_ERROR,

               "No such filter: '%s'\n", filt_name);

        return AVERROR(EINVAL);

    }



    ret = avfilter_open(filt_ctx, filt, inst_name);

    if (!*filt_ctx) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Error creating filter '%s'\n", filt_name);

        return ret;

    }



    if ((ret = avfilter_graph_add_filter(ctx, *filt_ctx)) < 0) {

        avfilter_free(*filt_ctx);

        return ret;

    }



    if (!strcmp(filt_name, "scale") && !strstr(args, "flags")) {

        snprintf(tmp_args, sizeof(tmp_args), "%s:%s",

                 args, ctx->scale_sws_opts);

        args = tmp_args;

    }



    if ((ret = avfilter_init_filter(*filt_ctx, args, NULL)) < 0) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Error initializing filter '%s' with args '%s'\n", filt_name, args);

        return ret;

    }



    return 0;

}
