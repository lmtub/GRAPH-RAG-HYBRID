static void qemu_opt_del(QemuOpt *opt)

{

    TAILQ_REMOVE(&opt->opts->head, opt, next);

    qemu_free((/* !const */ char*)opt->name);

    qemu_free((/* !const */ char*)opt->str);

    qemu_free(opt);

}
