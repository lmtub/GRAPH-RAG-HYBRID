static int parse_str(StringInputVisitor *siv, const char *name, Error **errp)

{

    char *str = (char *) siv->string;

    long long start, end;

    Range *cur;

    char *endptr;



    if (siv->ranges) {

        return 0;

    }



    do {

        errno = 0;

        start = strtoll(str, &endptr, 0);

        if (errno == 0 && endptr > str) {

            if (*endptr == '\0') {

                cur = g_malloc0(sizeof(*cur));

                cur->begin = start;

                cur->end = start + 1;

                siv->ranges = range_list_insert(siv->ranges, cur);

                cur = NULL;

                str = NULL;

            } else if (*endptr == '-') {

                str = endptr + 1;

                errno = 0;

                end = strtoll(str, &endptr, 0);

                if (errno == 0 && endptr > str && start <= end &&

                    (start > INT64_MAX - 65536 ||

                     end < start + 65536)) {

                    if (*endptr == '\0') {

                        cur = g_malloc0(sizeof(*cur));

                        cur->begin = start;

                        cur->end = end + 1;

                        siv->ranges = range_list_insert(siv->ranges, cur);

                        cur = NULL;

                        str = NULL;

                    } else if (*endptr == ',') {

                        str = endptr + 1;

                        cur = g_malloc0(sizeof(*cur));

                        cur->begin = start;

                        cur->end = end + 1;

                        siv->ranges = range_list_insert(siv->ranges, cur);

                        cur = NULL;

                    } else {

                        goto error;

                    }

                } else {

                    goto error;

                }

            } else if (*endptr == ',') {

                str = endptr + 1;

                cur = g_malloc0(sizeof(*cur));

                cur->begin = start;

                cur->end = start + 1;

                siv->ranges = range_list_insert(siv->ranges, cur);

                cur = NULL;

            } else {

                goto error;

            }

        } else {

            goto error;

        }

    } while (str);



    return 0;

error:

    g_list_foreach(siv->ranges, free_range, NULL);

    g_list_free(siv->ranges);

    siv->ranges = NULL;

    error_setg(errp, QERR_INVALID_PARAMETER_VALUE, name ? name : "null",

               "an int64 value or range");

    return -1;

}
