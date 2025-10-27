static int write_option(void *optctx, const OptionDef *po, const char *opt,

                        const char *arg)

{

    /* new-style options contain an offset into optctx, old-style address of

     * a global var*/

    void *dst = po->flags & (OPT_OFFSET | OPT_SPEC) ?

                (uint8_t *)optctx + po->u.off : po->u.dst_ptr;

    int *dstcount;



    if (po->flags & OPT_SPEC) {

        SpecifierOpt **so = dst;

        char *p = strchr(opt, ':');



        dstcount = (int *)(so + 1);

        *so = grow_array(*so, sizeof(**so), dstcount, *dstcount + 1);

        (*so)[*dstcount - 1].specifier = av_strdup(p ? p + 1 : "");

        dst = &(*so)[*dstcount - 1].u;

    }



    if (po->flags & OPT_STRING) {

        char *str;

        str = av_strdup(arg);

        av_freep(dst);

        *(char **)dst = str;

    } else if (po->flags & OPT_BOOL || po->flags & OPT_INT) {

        *(int *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT_MIN, INT_MAX);

    } else if (po->flags & OPT_INT64) {

        *(int64_t *)dst = parse_number_or_die(opt, arg, OPT_INT64, INT64_MIN, INT64_MAX);

    } else if (po->flags & OPT_TIME) {

        *(int64_t *)dst = parse_time_or_die(opt, arg, 1);

    } else if (po->flags & OPT_FLOAT) {

        *(float *)dst = parse_number_or_die(opt, arg, OPT_FLOAT, -INFINITY, INFINITY);

    } else if (po->flags & OPT_DOUBLE) {

        *(double *)dst = parse_number_or_die(opt, arg, OPT_DOUBLE, -INFINITY, INFINITY);

    } else if (po->u.func_arg) {

        int ret = po->u.func_arg(optctx, opt, arg);

        if (ret < 0) {

            av_log(NULL, AV_LOG_ERROR,

                   "Failed to set value '%s' for option '%s'\n", arg, opt);

            return ret;

        }

    }

    if (po->flags & OPT_EXIT)

        exit(0);



    return 0;

}
