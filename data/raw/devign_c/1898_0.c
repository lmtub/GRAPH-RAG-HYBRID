static int usb_parse(const char *cmdline)

{

    int r;

    r = usb_device_add(cmdline);

    if (r < 0) {

        fprintf(stderr, "qemu: could not add USB device '%s'\n", cmdline);

    }

    return r;

}
