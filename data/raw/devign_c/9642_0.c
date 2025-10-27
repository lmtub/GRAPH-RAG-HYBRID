static int proxy_utimensat(FsContext *s, V9fsPath *fs_path,

                           const struct timespec *buf)

{

    int retval;

    retval = v9fs_request(s->private, T_UTIME, NULL, "sqqqq",

                          fs_path,

                          buf[0].tv_sec, buf[0].tv_nsec,

                          buf[1].tv_sec, buf[1].tv_nsec);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
