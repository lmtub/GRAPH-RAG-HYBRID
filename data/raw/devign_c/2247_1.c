static int slirp_guestfwd(SlirpState *s, const char *config_str,

                          int legacy_format)

{

    struct in_addr server = { .s_addr = 0 };

    struct GuestFwd *fwd;

    const char *p;

    char buf[128];

    char *end;

    int port;



    p = config_str;

    if (legacy_format) {

        if (get_str_sep(buf, sizeof(buf), &p, ':') < 0) {

            goto fail_syntax;

        }

    } else {

        if (get_str_sep(buf, sizeof(buf), &p, ':') < 0) {

            goto fail_syntax;

        }

        if (strcmp(buf, "tcp") && buf[0] != '\0') {

            goto fail_syntax;

        }

        if (get_str_sep(buf, sizeof(buf), &p, ':') < 0) {

            goto fail_syntax;

        }

        if (buf[0] != '\0' && !inet_aton(buf, &server)) {

            goto fail_syntax;

        }

        if (get_str_sep(buf, sizeof(buf), &p, '-') < 0) {

            goto fail_syntax;

        }

    }

    port = strtol(buf, &end, 10);

    if (*end != '\0' || port < 1 || port > 65535) {

        goto fail_syntax;

    }



    fwd = g_malloc(sizeof(struct GuestFwd));

    snprintf(buf, sizeof(buf), "guestfwd.tcp.%d", port);



    if ((strlen(p) > 4) && !strncmp(p, "cmd:", 4)) {

        if (slirp_add_exec(s->slirp, 0, &p[4], &server, port) < 0) {

            error_report("conflicting/invalid host:port in guest forwarding "

                         "rule '%s'", config_str);

            g_free(fwd);

            return -1;

        }

    } else {

        fwd->hd = qemu_chr_new(buf, p, NULL);

        if (!fwd->hd) {

            error_report("could not open guest forwarding device '%s'", buf);

            g_free(fwd);

            return -1;

        }



        if (slirp_add_exec(s->slirp, 3, fwd->hd, &server, port) < 0) {

            error_report("conflicting/invalid host:port in guest forwarding "

                         "rule '%s'", config_str);

            g_free(fwd);

            return -1;

        }

        fwd->server = server;

        fwd->port = port;

        fwd->slirp = s->slirp;



        qemu_chr_fe_claim_no_fail(fwd->hd);

        qemu_chr_add_handlers(fwd->hd, guestfwd_can_read, guestfwd_read,

                              NULL, fwd);

    }

    return 0;



 fail_syntax:

    error_report("invalid guest forwarding rule '%s'", config_str);

    return -1;

}
