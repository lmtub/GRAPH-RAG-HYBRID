static int v9fs_synth_utimensat(FsContext *fs_ctx, V9fsPath *path,

                                const struct timespec *buf)

{

    errno = EPERM;

    return 0;

}
