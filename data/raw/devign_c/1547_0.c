static int qemu_rbd_set_conf(rados_t cluster, const char *conf)

{

    char *p, *buf;

    char name[RBD_MAX_CONF_NAME_SIZE];

    char value[RBD_MAX_CONF_VAL_SIZE];

    int ret = 0;



    buf = g_strdup(conf);

    p = buf;



    while (p) {

        ret = qemu_rbd_next_tok(name, sizeof(name), p,

                                '=', "conf option name", &p);

        if (ret < 0) {

            break;

        }



        if (!p) {

            error_report("conf option %s has no value", name);

            ret = -EINVAL;

            break;

        }



        ret = qemu_rbd_next_tok(value, sizeof(value), p,

                                ':', "conf option value", &p);

        if (ret < 0) {

            break;

        }



        if (strcmp(name, "conf")) {

            ret = rados_conf_set(cluster, name, value);

            if (ret < 0) {

                error_report("invalid conf option %s", name);

                ret = -EINVAL;

                break;

            }

        } else {

            ret = rados_conf_read_file(cluster, value);

            if (ret < 0) {

                error_report("error reading conf file %s", value);

                break;

            }

        }

    }



    g_free(buf);

    return ret;

}
