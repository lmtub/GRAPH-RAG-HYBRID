static ssize_t local_readlink(FsContext *fs_ctx, V9fsPath *fs_path,

                              char *buf, size_t bufsz)

{

    ssize_t tsize = -1;

    char *buffer;

    char *path = fs_path->data;



    if ((fs_ctx->export_flags & V9FS_SM_MAPPED) ||

        (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE)) {

        int fd;

        buffer = rpath(fs_ctx, path);

        fd = open(buffer, O_RDONLY | O_NOFOLLOW);

        g_free(buffer);

        if (fd == -1) {

            return -1;

        }

        do {

            tsize = read(fd, (void *)buf, bufsz);

        } while (tsize == -1 && errno == EINTR);

        close(fd);

    } else if ((fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

               (fs_ctx->export_flags & V9FS_SM_NONE)) {

        buffer = rpath(fs_ctx, path);

        tsize = readlink(buffer, buf, bufsz);

        g_free(buffer);

    }

    return tsize;

}
