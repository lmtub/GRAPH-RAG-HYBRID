int avfilter_init_str(AVFilterContext *filter, const char *args)

{

    return avfilter_init_filter(filter, args, NULL);

}
