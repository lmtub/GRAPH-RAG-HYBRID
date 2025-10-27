CharDriverState *qemu_chr_new_from_opts(QemuOpts *opts,

                                    void (*init)(struct CharDriverState *s),

                                    Error **errp)

{

    Error *local_err = NULL;

    CharDriver *cd;

    CharDriverState *chr;

    GSList *i;

    ChardevReturn *ret = NULL;

    ChardevBackend *backend;

    const char *id = qemu_opts_id(opts);

    char *bid = NULL;



    if (id == NULL) {

        error_setg(errp, "chardev: no id specified");

        goto err;

    }



    if (qemu_opt_get(opts, "backend") == NULL) {

        error_setg(errp, "chardev: \"%s\" missing backend",

                   qemu_opts_id(opts));

        goto err;

    }

    for (i = backends; i; i = i->next) {

        cd = i->data;



        if (strcmp(cd->name, qemu_opt_get(opts, "backend")) == 0) {

            break;

        }

    }

    if (i == NULL) {

        error_setg(errp, "chardev: backend \"%s\" not found",

                   qemu_opt_get(opts, "backend"));

        goto err;

    }



    backend = g_new0(ChardevBackend, 1);



    if (qemu_opt_get_bool(opts, "mux", 0)) {

        bid = g_strdup_printf("%s-base", id);

    }



    chr = NULL;

    backend->kind = cd->kind;

    if (cd->parse) {

        cd->parse(opts, backend, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            goto qapi_out;

        }

    }

    ret = qmp_chardev_add(bid ? bid : id, backend, errp);

    if (!ret) {

        goto qapi_out;

    }



    if (bid) {

        qapi_free_ChardevBackend(backend);

        qapi_free_ChardevReturn(ret);

        backend = g_new0(ChardevBackend, 1);

        backend->mux = g_new0(ChardevMux, 1);

        backend->kind = CHARDEV_BACKEND_KIND_MUX;

        backend->mux->chardev = g_strdup(bid);

        ret = qmp_chardev_add(id, backend, errp);

        if (!ret) {

            chr = qemu_chr_find(bid);

            qemu_chr_delete(chr);

            chr = NULL;

            goto qapi_out;

        }

    }



    chr = qemu_chr_find(id);

    chr->opts = opts;



qapi_out:

    qapi_free_ChardevBackend(backend);

    qapi_free_ChardevReturn(ret);

    g_free(bid);

    return chr;



err:

    qemu_opts_del(opts);

    return NULL;

}
