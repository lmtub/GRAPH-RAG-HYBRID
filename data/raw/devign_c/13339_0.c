int ff_parse_sample_rate(unsigned *ret, const char *arg, void *log_ctx)

{

    char *tail;

    double srate = av_strtod(arg, &tail);

    if (*tail || srate < 1 || (int)srate != srate) {

        av_log(log_ctx, AV_LOG_ERROR, "Invalid sample rate '%s'\n", arg);

        return AVERROR(EINVAL);

    }

    *ret = srate;

    return 0;

}
