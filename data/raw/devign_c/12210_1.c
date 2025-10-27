opts_type_size(Visitor *v, uint64_t *obj, const char *name, Error **errp)

{

    OptsVisitor *ov = DO_UPCAST(OptsVisitor, visitor, v);

    const QemuOpt *opt;

    int64_t val;

    char *endptr;



    opt = lookup_scalar(ov, name, errp);

    if (!opt) {

        return;

    }



    val = strtosz_suffix(opt->str ? opt->str : "", &endptr,

                         STRTOSZ_DEFSUFFIX_B);

    if (val != -1 && *endptr == '\0') {

        *obj = val;

        processed(ov, name);

        return;

    }

    error_set(errp, QERR_INVALID_PARAMETER_VALUE, opt->name,

              "a size value representible as a non-negative int64");

}
