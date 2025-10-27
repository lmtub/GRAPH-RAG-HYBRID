static int serial_parse(const char *devname)

{

    static int index = 0;

    char label[32];



    if (strcmp(devname, "none") == 0)

        return 0;

    if (index == MAX_SERIAL_PORTS) {

        fprintf(stderr, "qemu: too many serial ports\n");

        exit(1);

    }

    snprintf(label, sizeof(label), "serial%d", index);

    serial_hds[index] = qemu_chr_new(label, devname, NULL);

    if (!serial_hds[index]) {

        fprintf(stderr, "qemu: could not connect serial device"

                " to character backend '%s'\n", devname);

        return -1;

    }

    index++;

    return 0;

}
