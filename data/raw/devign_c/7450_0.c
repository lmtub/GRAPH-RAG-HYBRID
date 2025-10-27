int qemu_strtou64(const char *nptr, const char **endptr, int base,

                  uint64_t *result)

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

        /* FIXME This assumes uint64_t is unsigned long long */

        *result = strtoull(nptr, &p, base);

        /* Windows returns 1 for negative out-of-range values.  */

        if (errno == ERANGE) {

            *result = -1;

        }

        err = check_strtox_error(nptr, p, endptr, errno);

    }

    return err;

}
