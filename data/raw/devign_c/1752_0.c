static int qemu_rbd_parsename(const char *filename,

                              char *pool, int pool_len,

                              char *snap, int snap_len,

                              char *name, int name_len,

                              char *conf, int conf_len,

                              Error **errp)

{

    const char *start;

    char *p, *buf;

    int ret = 0;

    char *found_str;

    Error *local_err = NULL;



    if (!strstart(filename, "rbd:", &start)) {

        error_setg(errp, "File name must start with 'rbd:'");

        return -EINVAL;

    }



    buf = g_strdup(start);

    p = buf;

    *snap = '\0';

    *conf = '\0';



    found_str = qemu_rbd_next_tok(pool_len, p,

                                  '/', "pool name", &p, &local_err);

    if (local_err) {

        goto done;

    }

    if (!p) {

        ret = -EINVAL;

        error_setg(errp, "Pool name is required");

        goto done;

    }

    qemu_rbd_unescape(found_str);

    g_strlcpy(pool, found_str, pool_len);



    if (strchr(p, '@')) {

        found_str = qemu_rbd_next_tok(name_len, p,

                                      '@', "object name", &p, &local_err);

        if (local_err) {

            goto done;

        }

        qemu_rbd_unescape(found_str);

        g_strlcpy(name, found_str, name_len);



        found_str = qemu_rbd_next_tok(snap_len, p,

                                      ':', "snap name", &p, &local_err);

        if (local_err) {

            goto done;

        }

        qemu_rbd_unescape(found_str);

        g_strlcpy(snap, found_str, snap_len);

    } else {

        found_str = qemu_rbd_next_tok(name_len, p,

                                      ':', "object name", &p, &local_err);

        if (local_err) {

            goto done;

        }

        qemu_rbd_unescape(found_str);

        g_strlcpy(name, found_str, name_len);

    }

    if (!p) {

        goto done;

    }



    found_str = qemu_rbd_next_tok(conf_len, p,

                                  '\0', "configuration", &p, &local_err);

    if (local_err) {

        goto done;

    }

    g_strlcpy(conf, found_str, conf_len);



done:

    if (local_err) {

        ret = -EINVAL;

        error_propagate(errp, local_err);

    }

    g_free(buf);

    return ret;

}
