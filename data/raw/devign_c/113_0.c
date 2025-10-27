static void qemu_chr_parse_stdio(QemuOpts *opts, ChardevBackend *backend,

                                 Error **errp)

{

    ChardevStdio *stdio;



    stdio = backend->u.stdio = g_new0(ChardevStdio, 1);

    qemu_chr_parse_common(opts, qapi_ChardevStdio_base(stdio));

    stdio->has_signal = true;

    stdio->signal = qemu_opt_get_bool(opts, "signal", true);

}
