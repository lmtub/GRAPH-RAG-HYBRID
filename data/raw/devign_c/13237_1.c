QemuOpts *qemu_opts_parse(QemuOptsList *list, const char *params,

                          int permit_abbrev)

{

    const char *firstname;

    char value[1024], *id = NULL;

    const char *p;

    QemuOpts *opts;



    assert(!permit_abbrev || list->implied_opt_name);

    firstname = permit_abbrev ? list->implied_opt_name : NULL;



    if (strncmp(params, "id=", 3) == 0) {

        get_opt_value(value, sizeof(value), params+3);

        id = qemu_strdup(value);

    } else if ((p = strstr(params, ",id=")) != NULL) {

        get_opt_value(value, sizeof(value), p+4);

        id = qemu_strdup(value);

    }

    opts = qemu_opts_create(list, id, 1);

    if (opts == NULL)

        return NULL;



    if (qemu_opts_do_parse(opts, params, firstname) != 0) {

        qemu_opts_del(opts);

        return NULL;

    }



    return opts;

}
