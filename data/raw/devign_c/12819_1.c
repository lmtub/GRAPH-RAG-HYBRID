static int check_strtox_error(const char **next, char *endptr,

                              int err)

{

    if (!next && *endptr) {

        return -EINVAL;

    }

    if (next) {

        *next = endptr;

    }

    return -err;

}
