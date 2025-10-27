static bool qemu_opt_get_bool_helper(QemuOpts *opts, const char *name,

                                     bool defval, bool del)

{

    QemuOpt *opt = qemu_opt_find(opts, name);

    bool ret = defval;



    if (opt == NULL) {

        const QemuOptDesc *desc = find_desc_by_name(opts->list->desc, name);

        if (desc && desc->def_value_str) {

            parse_option_bool(name, desc->def_value_str, &ret, &error_abort);

        }

        return ret;

    }

    assert(opt->desc && opt->desc->type == QEMU_OPT_BOOL);

    ret = opt->value.boolean;

    if (del) {

        qemu_opt_del_all(opts, name);

    }

    return ret;

}
