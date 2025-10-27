int qemu_strtol(const char *nptr, const char **endptr, int base,

                long *result)

{

    char *ep;

    int err = 0;

    if (!nptr) {

        if (endptr) {

            *endptr = nptr;

        }

        err = -EINVAL;

    } else {

        errno = 0;

        *result = strtol(nptr, &ep, base);

        err = check_strtox_error(nptr, ep, endptr, errno);

    }

    return err;

}
