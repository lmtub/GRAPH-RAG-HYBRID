static int check_strtox_error(const char *p, char *endptr, const char **next,

                              int err)

{

    if (err == 0 && endptr == p) {

        err = EINVAL;

    }

    if (!next && *endptr) {

        return -EINVAL;

    }

    if (next) {

        *next = endptr;

    }

    return -err;

}
