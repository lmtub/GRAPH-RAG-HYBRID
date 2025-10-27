CharDriverState *qemu_chr_new_from_opts(QemuOpts *opts,

                                    void (*init)(struct CharDriverState *s))

{

    CharDriverState *chr;

    int i;

    int ret;



    if (qemu_opts_id(opts) == NULL) {

        fprintf(stderr, "chardev: no id specified\n");

        return NULL;

    }



    if (qemu_opt_get(opts, "backend") == NULL) {

        fprintf(stderr, "chardev: \"%s\" missing backend\n",

                qemu_opts_id(opts));

        return NULL;

    }

    for (i = 0; i < ARRAY_SIZE(backend_table); i++) {

        if (strcmp(backend_table[i].name, qemu_opt_get(opts, "backend")) == 0)

            break;

    }

    if (i == ARRAY_SIZE(backend_table)) {

        fprintf(stderr, "chardev: backend \"%s\" not found\n",

                qemu_opt_get(opts, "backend"));

        return NULL;

    }



    ret = backend_table[i].open(opts, &chr);

    if (ret < 0) {

        fprintf(stderr, "chardev: opening backend \"%s\" failed: %s\n",

                qemu_opt_get(opts, "backend"), strerror(-ret));

        return NULL;

    }



    if (!chr->filename)

        chr->filename = g_strdup(qemu_opt_get(opts, "backend"));

    chr->init = init;

    QTAILQ_INSERT_TAIL(&chardevs, chr, next);



    if (qemu_opt_get_bool(opts, "mux", 0)) {

        CharDriverState *base = chr;

        int len = strlen(qemu_opts_id(opts)) + 6;

        base->label = g_malloc(len);

        snprintf(base->label, len, "%s-base", qemu_opts_id(opts));

        chr = qemu_chr_open_mux(base);

        chr->filename = base->filename;

        chr->avail_connections = MAX_MUX;

        QTAILQ_INSERT_TAIL(&chardevs, chr, next);

    } else {

        chr->avail_connections = 1;

    }

    chr->label = g_strdup(qemu_opts_id(opts));

    return chr;

}
