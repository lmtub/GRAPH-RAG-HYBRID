static uint64_t qemu_opt_get_size_helper(QemuOpts *opts, const char *name,

                                         uint64_t defval, bool del)

{

    QemuOpt *opt = qemu_opt_find(opts, name);

    uint64_t ret = defval;



    if (opt == NULL) {

        const QemuOptDesc *desc = find_desc_by_name(opts->list->desc, name);

        if (desc && desc->def_value_str) {

            parse_option_size(name, desc->def_value_str, &ret, &error_abort);

        }

        return ret;

    }

    assert(opt->desc && opt->desc->type == QEMU_OPT_SIZE);

    ret = opt->value.uint;

    if (del) {

        qemu_opt_del_all(opts, name);

    }

    return ret;

}
