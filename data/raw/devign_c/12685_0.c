static QemuOpt *qemu_opt_find(QemuOpts *opts, const char *name)

{

    QemuOpt *opt;



    TAILQ_FOREACH(opt, &opts->head, next) {

        if (strcmp(opt->name, name) != 0)

            continue;

        return opt;

    }

    return NULL;

}
