static int opt_recording_timestamp(void *optctx, const char *opt, const char *arg)

{

    OptionsContext *o = optctx;

    char buf[128];

    int64_t recording_timestamp = parse_time_or_die(opt, arg, 0) / 1E6;

    struct tm time = *gmtime((time_t*)&recording_timestamp);

    strftime(buf, sizeof(buf), "creation_time=%FT%T%z", &time);

    parse_option(o, "metadata", buf, options);



    av_log(NULL, AV_LOG_WARNING, "%s is deprecated, set the 'creation_time' metadata "

                                 "tag instead.\n", opt);

    return 0;

}
