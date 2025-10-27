static int v9fs_synth_mkdir(FsContext *fs_ctx, V9fsPath *path,

                       const char *buf, FsCred *credp)

{

    errno = EPERM;

    return -1;

}
