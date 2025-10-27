static int parallel_parse(const char *devname)

{

    static int index = 0;

    char label[32];



    if (strcmp(devname, "none") == 0)

        return 0;

    if (index == MAX_PARALLEL_PORTS) {

        fprintf(stderr, "qemu: too many parallel ports\n");

        exit(1);

    }

    snprintf(label, sizeof(label), "parallel%d", index);

    parallel_hds[index] = qemu_chr_new(label, devname, NULL);

    if (!parallel_hds[index]) {

        fprintf(stderr, "qemu: could not connect parallel device"

                " to character backend '%s'\n", devname);

        return -1;

    }

    index++;

    return 0;

}
