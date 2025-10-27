static void qemu_chr_parse_serial(QemuOpts *opts, ChardevBackend *backend,

                                  Error **errp)

{

    const char *device = qemu_opt_get(opts, "path");



    if (device == NULL) {

        error_setg(errp, "chardev: serial/tty: no device path given");

        return;

    }

    backend->serial = g_new0(ChardevHostdev, 1);

    backend->serial->device = g_strdup(device);

}
