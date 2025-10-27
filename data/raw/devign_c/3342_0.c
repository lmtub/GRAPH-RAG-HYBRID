static int local_rename(FsContext *ctx, const char *oldpath,

                        const char *newpath)

{

    char *tmp;

    int err;



    tmp = qemu_strdup(rpath(ctx, oldpath));

    if (tmp == NULL) {

        return -1;

    }



    err = rename(tmp, rpath(ctx, newpath));

    if (err == -1) {

        int serrno = errno;

        qemu_free(tmp);

        errno = serrno;

    } else {

        qemu_free(tmp);

    }



    return err;



}
