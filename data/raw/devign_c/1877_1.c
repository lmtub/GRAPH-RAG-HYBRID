int qemu_strtol(const char *nptr, const char **endptr, int base,

                long *result)

{

    char *p;

    int err = 0;

    if (!nptr) {

        if (endptr) {

            *endptr = nptr;

        }

        err = -EINVAL;

    } else {

        errno = 0;

        *result = strtol(nptr, &p, base);

        err = check_strtox_error(endptr, p, errno);

    }

    return err;

}
