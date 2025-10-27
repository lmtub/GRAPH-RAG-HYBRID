static int proxy_lstat(FsContext *fs_ctx, V9fsPath *fs_path, struct stat *stbuf)

{

    int retval;

    retval = v9fs_request(fs_ctx->private, T_LSTAT, stbuf, "s", fs_path);

    if (retval < 0) {

        errno = -retval;

        return -1;

    }

    return retval;

}
