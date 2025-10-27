static int qemu_rbd_set_conf(rados_t cluster, const char *conf,

                             bool only_read_conf_file,

                             Error **errp)

{

    char *p, *buf;

    char *name;

    char *value;

    Error *local_err = NULL;

    int ret = 0;



    buf = g_strdup(conf);

    p = buf;



    while (p) {

        name = qemu_rbd_next_tok(RBD_MAX_CONF_NAME_SIZE, p,

                                 '=', "conf option name", &p, &local_err);

        if (local_err) {

            break;

        }

        qemu_rbd_unescape(name);



        if (!p) {

            error_setg(errp, "conf option %s has no value", name);

            ret = -EINVAL;

            break;

        }



        value = qemu_rbd_next_tok(RBD_MAX_CONF_VAL_SIZE, p,

                                  ':', "conf option value", &p, &local_err);

        if (local_err) {

            break;

        }

        qemu_rbd_unescape(value);



        if (strcmp(name, "conf") == 0) {

            /* read the conf file alone, so it doesn't override more

               specific settings for a particular device */

            if (only_read_conf_file) {

                ret = rados_conf_read_file(cluster, value);

                if (ret < 0) {

                    error_setg_errno(errp, -ret, "error reading conf file %s",

                                     value);

                    break;

                }

            }

        } else if (strcmp(name, "id") == 0) {

            /* ignore, this is parsed by qemu_rbd_parse_clientname() */

        } else if (!only_read_conf_file) {

            ret = rados_conf_set(cluster, name, value);

            if (ret < 0) {

                error_setg_errno(errp, -ret, "invalid conf option %s", name);

                ret = -EINVAL;

                break;

            }

        }

    }



    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

    }

    g_free(buf);

    return ret;

}
