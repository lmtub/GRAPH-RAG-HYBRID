static int virtcon_parse(const char *devname)

{

    QemuOptsList *device = qemu_find_opts("device");

    static int index = 0;

    char label[32];

    QemuOpts *bus_opts, *dev_opts;



    if (strcmp(devname, "none") == 0)

        return 0;

    if (index == MAX_VIRTIO_CONSOLES) {

        fprintf(stderr, "qemu: too many virtio consoles\n");

        exit(1);

    }



    bus_opts = qemu_opts_create(device, NULL, 0, &error_abort);

    if (arch_type == QEMU_ARCH_S390X) {

        qemu_opt_set(bus_opts, "driver", "virtio-serial-s390", &error_abort);

    } else {

        qemu_opt_set(bus_opts, "driver", "virtio-serial-pci", &error_abort);

    }



    dev_opts = qemu_opts_create(device, NULL, 0, &error_abort);

    qemu_opt_set(dev_opts, "driver", "virtconsole", &error_abort);



    snprintf(label, sizeof(label), "virtcon%d", index);

    virtcon_hds[index] = qemu_chr_new(label, devname, NULL);

    if (!virtcon_hds[index]) {

        fprintf(stderr, "qemu: could not connect virtio console"

                " to character backend '%s'\n", devname);

        return -1;

    }

    qemu_opt_set(dev_opts, "chardev", label, &error_abort);



    index++;

    return 0;

}
