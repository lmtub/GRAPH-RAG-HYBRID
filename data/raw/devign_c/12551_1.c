int unix_listen(const char *str, char *ostr, int olen)

{

    QemuOpts *opts;

    char *path, *optstr;

    int sock, len;



    opts = qemu_opts_create(&dummy_opts, NULL, 0);



    optstr = strchr(str, ',');

    if (optstr) {

        len = optstr - str;

        if (len) {

            path = g_malloc(len+1);

            snprintf(path, len+1, "%.*s", len, str);

            qemu_opt_set(opts, "path", path);

            g_free(path);

        }

    } else {

        qemu_opt_set(opts, "path", str);

    }



    sock = unix_listen_opts(opts);



    if (sock != -1 && ostr)

        snprintf(ostr, olen, "%s%s", qemu_opt_get(opts, "path"), optstr ? optstr : "");

    qemu_opts_del(opts);

    return sock;

}
