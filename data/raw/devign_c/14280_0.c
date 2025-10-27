bool has_help_option(const char *param)

{

    size_t buflen = strlen(param) + 1;

    char *buf = g_malloc(buflen);

    const char *p = param;

    bool result = false;



    while (*p) {

        p = get_opt_value(buf, buflen, p);

        if (*p) {

            p++;

        }



        if (is_help_option(buf)) {

            result = true;

            goto out;

        }

    }



out:

    free(buf);

    return result;

}
