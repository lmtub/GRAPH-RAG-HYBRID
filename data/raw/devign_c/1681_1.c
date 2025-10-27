static int opt_sameq(void *optctx, const char *opt, const char *arg)

{

    av_log(NULL, AV_LOG_WARNING, "Ignoring option '%s'\n", opt);

    return 0;

}
