static void qemu_chr_parse_common(QemuOpts *opts, ChardevCommon *backend)

{

    const char *logfile = qemu_opt_get(opts, "logfile");



    backend->has_logfile = logfile != NULL;

    backend->logfile = logfile ? g_strdup(logfile) : NULL;



    backend->has_logappend = true;

    backend->logappend = qemu_opt_get_bool(opts, "logappend", false);

}
