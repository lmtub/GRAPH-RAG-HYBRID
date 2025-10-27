int qemu_opts_do_parse(QemuOpts *opts, const char *params, const char *firstname)

{

    Error *err = NULL;



    opts_do_parse(opts, params, firstname, false, &err);

    if (err) {

        qerror_report_err(err);

        error_free(err);

        return -1;

    }

    return 0;

}
