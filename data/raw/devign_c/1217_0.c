static ssize_t v9fs_synth_lgetxattr(FsContext *ctx, V9fsPath *path,

                                    const char *name, void *value, size_t size)

{

    errno = ENOTSUP;

    return -1;

}
