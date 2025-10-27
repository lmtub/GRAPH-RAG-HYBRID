void avfilter_uninit(void)

{

    memset(registered_avfilters, 0, sizeof(registered_avfilters));

    next_registered_avfilter_idx = 0;

}
