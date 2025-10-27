static void monitor_parse(const char *optarg, const char *mode, bool pretty)

{

    static int monitor_device_index = 0;

    Error *local_err = NULL;

    QemuOpts *opts;

    const char *p;

    char label[32];

    int def = 0;



    if (strstart(optarg, "chardev:", &p)) {

        snprintf(label, sizeof(label), "%s", p);

    } else {

        snprintf(label, sizeof(label), "compat_monitor%d",

                 monitor_device_index);

        if (monitor_device_index == 0) {

            def = 1;

        }

        opts = qemu_chr_parse_compat(label, optarg);

        if (!opts) {

            fprintf(stderr, "parse error: %s\n", optarg);

            exit(1);

        }

    }



    opts = qemu_opts_create(qemu_find_opts("mon"), label, 1, &local_err);

    if (!opts) {

        error_report_err(local_err);

        exit(1);

    }

    qemu_opt_set(opts, "mode", mode, &error_abort);

    qemu_opt_set(opts, "chardev", label, &error_abort);

    qemu_opt_set_bool(opts, "pretty", pretty, &error_abort);

    if (def)

        qemu_opt_set(opts, "default", "on", &error_abort);

    monitor_device_index++;

}
