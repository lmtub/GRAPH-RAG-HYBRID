void net_slirp_hostfwd_remove(Monitor *mon, const QDict *qdict)

{

    struct in_addr host_addr = { .s_addr = INADDR_ANY };

    int host_port;

    char buf[256] = "";

    const char *src_str, *p;

    SlirpState *s;

    int is_udp = 0;

    int err;

    const char *arg1 = qdict_get_str(qdict, "arg1");

    const char *arg2 = qdict_get_try_str(qdict, "arg2");

    const char *arg3 = qdict_get_try_str(qdict, "arg3");



    if (arg2) {

        s = slirp_lookup(mon, arg1, arg2);

        src_str = arg3;

    } else {

        s = slirp_lookup(mon, NULL, NULL);

        src_str = arg1;

    }

    if (!s) {

        return;

    }



    if (!src_str || !src_str[0])

        goto fail_syntax;



    p = src_str;

    get_str_sep(buf, sizeof(buf), &p, ':');



    if (!strcmp(buf, "tcp") || buf[0] == '\0') {

        is_udp = 0;

    } else if (!strcmp(buf, "udp")) {

        is_udp = 1;

    } else {

        goto fail_syntax;

    }



    if (get_str_sep(buf, sizeof(buf), &p, ':') < 0) {

        goto fail_syntax;

    }

    if (buf[0] != '\0' && !inet_aton(buf, &host_addr)) {

        goto fail_syntax;

    }



    host_port = atoi(p);



    err = slirp_remove_hostfwd(QTAILQ_FIRST(&slirp_stacks)->slirp, is_udp,

                               host_addr, host_port);



    monitor_printf(mon, "host forwarding rule for %s %s\n", src_str,

                   err ? "removed" : "not found");

    return;



 fail_syntax:

    monitor_printf(mon, "invalid format\n");

}
