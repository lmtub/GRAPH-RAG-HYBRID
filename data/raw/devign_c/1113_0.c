static void qemu_rbd_parse_filename(const char *filename, QDict *options,

                                    Error **errp)

{

    const char *start;

    char *p, *buf, *keypairs;

    char *found_str;

    size_t max_keypair_size;

    Error *local_err = NULL;



    if (!strstart(filename, "rbd:", &start)) {

        error_setg(errp, "File name must start with 'rbd:'");

        return;

    }



    max_keypair_size = strlen(start) + 1;

    buf = g_strdup(start);

    keypairs = g_malloc0(max_keypair_size);

    p = buf;



    found_str = qemu_rbd_next_tok(RBD_MAX_POOL_NAME_SIZE, p,

                                  '/', "pool name", &p, &local_err);

    if (local_err) {

        goto done;

    }

    if (!p) {

        error_setg(errp, "Pool name is required");

        goto done;

    }

    qemu_rbd_unescape(found_str);

    qdict_put(options, "pool", qstring_from_str(found_str));



    if (strchr(p, '@')) {

        found_str = qemu_rbd_next_tok(RBD_MAX_IMAGE_NAME_SIZE, p,

                                      '@', "object name", &p, &local_err);

        if (local_err) {

            goto done;

        }

        qemu_rbd_unescape(found_str);

        qdict_put(options, "image", qstring_from_str(found_str));



        found_str = qemu_rbd_next_tok(RBD_MAX_SNAP_NAME_SIZE, p,

                                      ':', "snap name", &p, &local_err);

        if (local_err) {

            goto done;

        }

        qemu_rbd_unescape(found_str);

        qdict_put(options, "snapshot", qstring_from_str(found_str));

    } else {

        found_str = qemu_rbd_next_tok(RBD_MAX_IMAGE_NAME_SIZE, p,

                                      ':', "object name", &p, &local_err);

        if (local_err) {

            goto done;

        }

        qemu_rbd_unescape(found_str);

        qdict_put(options, "image", qstring_from_str(found_str));

    }

    if (!p) {

        goto done;

    }



    found_str = qemu_rbd_next_tok(RBD_MAX_CONF_NAME_SIZE, p,

                                  '\0', "configuration", &p, &local_err);

    if (local_err) {

        goto done;

    }



    p = found_str;



    /* The following are essentially all key/value pairs, and we treat

     * 'id' and 'conf' a bit special.  Key/value pairs may be in any order. */

    while (p) {

        char *name, *value;

        name = qemu_rbd_next_tok(RBD_MAX_CONF_NAME_SIZE, p,

                                 '=', "conf option name", &p, &local_err);

        if (local_err) {

            break;

        }



        if (!p) {

            error_setg(errp, "conf option %s has no value", name);

            break;

        }



        qemu_rbd_unescape(name);



        value = qemu_rbd_next_tok(RBD_MAX_CONF_VAL_SIZE, p,

                                  ':', "conf option value", &p, &local_err);

        if (local_err) {

            break;

        }

        qemu_rbd_unescape(value);



        if (!strcmp(name, "conf")) {

            qdict_put(options, "conf", qstring_from_str(value));

        } else if (!strcmp(name, "id")) {

            qdict_put(options, "user" , qstring_from_str(value));

        } else {

            /* FIXME: This is pretty ugly, and not the right way to do this.

             *        These should be contained in a structure, and then

             *        passed explicitly as individual key/value pairs to

             *        rados.  Consider this legacy code that needs to be

             *        updated. */

            char *tmp = g_malloc0(max_keypair_size);

            /* only use a delimiter if it is not the first keypair found */

            /* These are sets of unknown key/value pairs we'll pass along

             * to ceph */

            if (keypairs[0]) {

                snprintf(tmp, max_keypair_size, ":%s=%s", name, value);

                pstrcat(keypairs, max_keypair_size, tmp);

            } else {

                snprintf(keypairs, max_keypair_size, "%s=%s", name, value);

            }

            g_free(tmp);

        }

    }



    if (keypairs[0]) {

        qdict_put(options, "keyvalue-pairs", qstring_from_str(keypairs));

    }





done:

    if (local_err) {

        error_propagate(errp, local_err);

    }

    g_free(buf);

    g_free(keypairs);

    return;

}
