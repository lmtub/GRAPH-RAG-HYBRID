static int v9fs_synth_chown(FsContext *fs_ctx, V9fsPath *path, FsCred *credp)

{

    errno = EPERM;

    return -1;

}
