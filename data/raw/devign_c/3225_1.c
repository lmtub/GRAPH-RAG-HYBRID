static int fchmodat_nofollow(int dirfd, const char *name, mode_t mode)

{

    int fd, ret;



    /* FIXME: this should be handled with fchmodat(AT_SYMLINK_NOFOLLOW).

     * Unfortunately, the linux kernel doesn't implement it yet. As an

     * alternative, let's open the file and use fchmod() instead. This

     * may fail depending on the permissions of the file, but it is the

     * best we can do to avoid TOCTTOU. We first try to open read-only

     * in case name points to a directory. If that fails, we try write-only

     * in case name doesn't point to a directory.

     */

    fd = openat_file(dirfd, name, O_RDONLY, 0);

    if (fd == -1) {

        /* In case the file is writable-only and isn't a directory. */

        if (errno == EACCES) {

            fd = openat_file(dirfd, name, O_WRONLY, 0);

        }

        if (fd == -1 && errno == EISDIR) {

            errno = EACCES;

        }

    }

    if (fd == -1) {

        return -1;

    }

    ret = fchmod(fd, mode);

    close_preserve_errno(fd);

    return ret;

}
