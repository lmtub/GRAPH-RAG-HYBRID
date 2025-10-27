static int v9fs_synth_link(FsContext *fs_ctx, V9fsPath *oldpath,

                           V9fsPath *newpath, const char *buf)

{

    errno = EPERM;

    return -1;

}
