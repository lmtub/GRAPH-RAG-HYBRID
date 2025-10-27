opts_type_size(Visitor *v, const char *name, uint64_t *obj, Error **errp)

{

    OptsVisitor *ov = to_ov(v);

    const QemuOpt *opt;

    int64_t val;



    opt = lookup_scalar(ov, name, errp);

    if (!opt) {

        return;

    }



    val = qemu_strtosz(opt->str ? opt->str : "", NULL);

    if (val < 0) {

        error_setg(errp, QERR_INVALID_PARAMETER_VALUE, opt->name,

                   "a size value representible as a non-negative int64");

        return;

    }



    *obj = val;

    processed(ov, name);

}
