static int net_vhost_chardev_opts(void *opaque,

                                  const char *name, const char *value,

                                  Error **errp)

{

    VhostUserChardevProps *props = opaque;



    if (strcmp(name, "backend") == 0 && strcmp(value, "socket") == 0) {

        props->is_socket = true;

    } else if (strcmp(name, "path") == 0) {

        props->is_unix = true;

    } else if (strcmp(name, "server") == 0) {

    } else {

        error_setg(errp,

                   "vhost-user does not support a chardev with option %s=%s",

                   name, value);

        return -1;

    }

    return 0;

}
