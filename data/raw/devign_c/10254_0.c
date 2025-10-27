static ssize_t v9fs_synth_readlink(FsContext *fs_ctx, V9fsPath *path,

                                   char *buf, size_t bufsz)

{

    errno = ENOSYS;

    return -1;

}
