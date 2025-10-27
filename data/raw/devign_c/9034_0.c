static int check_strtox_error(const char *nptr, char *ep,

                              const char **endptr, int libc_errno)

{

    if (libc_errno == 0 && ep == nptr) {

        libc_errno = EINVAL;

    }

    if (!endptr && *ep) {

        return -EINVAL;

    }

    if (endptr) {

        *endptr = ep;

    }

    return -libc_errno;

}
