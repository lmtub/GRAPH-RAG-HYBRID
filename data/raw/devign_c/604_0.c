int check_params(char *buf, int buf_size,

                 const char * const *params, const char *str)

{

    const char *p;

    int i;



    p = str;

    while (*p != '\0') {

        p = get_opt_name(buf, buf_size, p, '=');

        if (*p != '=') {

            return -1;

        }

        p++;

        for (i = 0; params[i] != NULL; i++) {

            if (!strcmp(params[i], buf)) {

                break;

            }

        }

        if (params[i] == NULL) {

            return -1;

        }

        p = get_opt_value(NULL, 0, p);

        if (*p != ',') {

            break;

        }

        p++;

    }

    return 0;

}
