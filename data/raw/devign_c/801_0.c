static int proxy_opendir(FsContext *ctx,

                         V9fsPath *fs_path, V9fsFidOpenState *fs)

{

    int serrno, fd;



    fs->dir = NULL;

    fd = v9fs_request(ctx->private, T_OPEN, NULL, "sd", fs_path, O_DIRECTORY);

    if (fd < 0) {

        errno = -fd;

        return -1;

    }

    fs->dir = fdopendir(fd);

    if (!fs->dir) {

        serrno = errno;

        close(fd);

        errno = serrno;

        return -1;

    }

    return 0;

}
