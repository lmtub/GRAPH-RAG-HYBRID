static int sclp_parse(const char *devname)

{

    QemuOptsList *device = qemu_find_opts("device");

    static int index = 0;

    char label[32];

    QemuOpts *dev_opts;



    if (strcmp(devname, "none") == 0) {

        return 0;

    }

    if (index == MAX_SCLP_CONSOLES) {

        fprintf(stderr, "qemu: too many sclp consoles\n");

        exit(1);

    }



    assert(arch_type == QEMU_ARCH_S390X);



    dev_opts = qemu_opts_create(device, NULL, 0, NULL);

    qemu_opt_set(dev_opts, "driver", "sclpconsole", &error_abort);



    snprintf(label, sizeof(label), "sclpcon%d", index);

    sclp_hds[index] = qemu_chr_new(label, devname, NULL);

    if (!sclp_hds[index]) {

        fprintf(stderr, "qemu: could not connect sclp console"

                " to character backend '%s'\n", devname);

        return -1;

    }

    qemu_opt_set(dev_opts, "chardev", label, &error_abort);



    index++;

    return 0;

}
