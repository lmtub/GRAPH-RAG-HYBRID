opts_type_uint64(Visitor *v, uint64_t *obj, const char *name, Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);

    const QemuOpt *opt;

    const char *str;

    unsigned long long val;

    char *endptr;



    if (ov->list_mode == LM_UNSIGNED_INTERVAL) {

        *obj = ov->range_next.u;

        return;

    }



    opt = lookup_scalar(ov, name, errp);

    if (!opt) {

        return;

    }

    str = opt->str;



    /* we've gotten past lookup_scalar() */

    assert(ov->list_mode == LM_NONE || ov->list_mode == LM_IN_PROGRESS);



    if (parse_uint(str, &val, &endptr, 0) == 0 && val <= UINT64_MAX) {

        if (*endptr == '\0') {

            *obj = val;

            processed(ov, name);

            return;

        }

        if (*endptr == '-' && ov->list_mode == LM_IN_PROGRESS) {

            unsigned long long val2;



            str = endptr + 1;

            if (parse_uint_full(str, &val2, 0) == 0 &&

                val2 <= UINT64_MAX && val <= val2) {

                ov->range_next.u = val;

                ov->range_limit.u = val2;

                ov->list_mode = LM_UNSIGNED_INTERVAL;



                /* as if entering on the top */

                *obj = ov->range_next.u;

                return;

            }

        }

    }

    error_set(errp, QERR_INVALID_PARAMETER_VALUE, opt->name,

              (ov->list_mode == LM_NONE) ? "a uint64 value" :

                                           "a uint64 value or range");

}
