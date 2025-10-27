int64_t parse_time_or_die(const char *context, const char *timestr,

                          int is_duration)

{

    int64_t us;

    if (av_parse_time(&us, timestr, is_duration) < 0) {

        av_log(NULL, AV_LOG_FATAL, "Invalid %s specification for %s: %s\n",

               is_duration ? "duration" : "date", context, timestr);

        exit(1);

    }

    return us;

}
