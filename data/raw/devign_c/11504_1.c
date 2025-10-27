static void parse_cmdline(const char *cmdline,

                          int *pnb_args, char **args)

{

    const char *p;

    int nb_args, ret;

    char buf[1024];



    p = cmdline;

    nb_args = 0;

    for (;;) {

        while (qemu_isspace(*p)) {

            p++;

        }

        if (*p == '\0') {

            break;

        }

        if (nb_args >= MAX_ARGS) {

            break;

        }

        ret = get_str(buf, sizeof(buf), &p);

        args[nb_args] = g_strdup(buf);

        nb_args++;

        if (ret < 0) {

            break;

        }

    }

    *pnb_args = nb_args;

}
