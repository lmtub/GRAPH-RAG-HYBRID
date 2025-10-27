static int v9fs_synth_open2(FsContext *fs_ctx, V9fsPath *dir_path,

                            const char *name, int flags,

                            FsCred *credp, V9fsFidOpenState *fs)

{

    errno = ENOSYS;

    return -1;

}
