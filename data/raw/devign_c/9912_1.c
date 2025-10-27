CharDriverState *qemu_chr_open_opts(QemuOpts *opts,

                                    void (*init)(struct CharDriverState *s))

{

    CharDriverState *chr;

    int i;



    if (qemu_opts_id(opts) == NULL) {

        fprintf(stderr, "chardev: no id specified\n");










    for (i = 0; i < ARRAY_SIZE(backend_table); i++) {

        if (strcmp(backend_table[i].name, qemu_opt_get(opts, "backend")) == 0)

            break;


    if (i == ARRAY_SIZE(backend_table)) {

        fprintf(stderr, "chardev: backend \"%s\" not found\n",

                qemu_opt_get(opts, "backend"));





    chr = backend_table[i].open(opts);

    if (!chr) {

        fprintf(stderr, "chardev: opening backend \"%s\" failed\n",

                qemu_opt_get(opts, "backend"));





    if (!chr->filename)

        chr->filename = qemu_strdup(qemu_opt_get(opts, "backend"));

    chr->init = init;

    QTAILQ_INSERT_TAIL(&chardevs, chr, next);



    if (qemu_opt_get_bool(opts, "mux", 0)) {

        CharDriverState *base = chr;

        int len = strlen(qemu_opts_id(opts)) + 6;

        base->label = qemu_malloc(len);

        snprintf(base->label, len, "%s-base", qemu_opts_id(opts));

        chr = qemu_chr_open_mux(base);

        chr->filename = base->filename;

        QTAILQ_INSERT_TAIL(&chardevs, chr, next);


    chr->label = qemu_strdup(qemu_opts_id(opts));

    return chr;
