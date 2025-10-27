CharDriverState *qemu_chr_new(const char *label, const char *filename, void (*init)(struct CharDriverState *s))

{

    const char *p;

    CharDriverState *chr;

    QemuOpts *opts;

    Error *err = NULL;



    if (strstart(filename, "chardev:", &p)) {

        return qemu_chr_find(p);

    }



    opts = qemu_chr_parse_compat(label, filename);

    if (!opts)

        return NULL;



    chr = qemu_chr_new_from_opts(opts, init, &err);

    if (err) {

        error_report_err(err);

    }

    if (chr && qemu_opt_get_bool(opts, "mux", 0)) {

        qemu_chr_fe_claim_no_fail(chr);

        monitor_init(chr, MONITOR_USE_READLINE);

    }

    return chr;

}
