int avfilter_register(AVFilter *filter)

{

    if (next_registered_avfilter_idx == MAX_REGISTERED_AVFILTERS_NB)

        return -1;



    registered_avfilters[next_registered_avfilter_idx++] = filter;

    return 0;

}
