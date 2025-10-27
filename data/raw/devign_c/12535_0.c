static void qemu_chr_parse_mux(QemuOpts *opts, ChardevBackend *backend,

                               Error **errp)

{

    const char *chardev = qemu_opt_get(opts, "chardev");

    ChardevMux *mux;



    if (chardev == NULL) {

        error_setg(errp, "chardev: mux: no chardev given");

        return;

    }

    mux = backend->u.mux = g_new0(ChardevMux, 1);

    qemu_chr_parse_common(opts, qapi_ChardevMux_base(mux));

    mux->chardev = g_strdup(chardev);

}
