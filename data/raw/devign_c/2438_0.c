static int opt_debug(void *optctx, const char *opt, const char *arg)

{

    av_log_set_level(99);

    debug = parse_number_or_die(opt, arg, OPT_INT64, 0, INT_MAX);

    return 0;

}
