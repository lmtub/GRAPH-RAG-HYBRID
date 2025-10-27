static int mon_init_func(QemuOpts *opts, void *opaque)

{

    CharDriverState *chr;

    const char *chardev;

    const char *mode;

    int flags;



    mode = qemu_opt_get(opts, "mode");

    if (mode == NULL) {

        mode = "readline";

    }

    if (strcmp(mode, "readline") == 0) {

        flags = MONITOR_USE_READLINE;

    } else if (strcmp(mode, "control") == 0) {

        flags = MONITOR_USE_CONTROL;

    } else {

        fprintf(stderr, "unknown monitor mode \"%s\"\n", mode);

        exit(1);

    }



    if (qemu_opt_get_bool(opts, "pretty", 0))

        flags |= MONITOR_USE_PRETTY;



    if (qemu_opt_get_bool(opts, "default", 0))

        flags |= MONITOR_IS_DEFAULT;



    chardev = qemu_opt_get(opts, "chardev");

    chr = qemu_chr_find(chardev);

    if (chr == NULL) {

        fprintf(stderr, "chardev \"%s\" not found\n", chardev);

        exit(1);

    }




    monitor_init(chr, flags);

    return 0;

}