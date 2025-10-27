static int v9fs_synth_unlinkat(FsContext *ctx, V9fsPath *dir,

                               const char *name, int flags)

{

    errno = EPERM;

    return -1;

}
