static int v9fs_synth_renameat(FsContext *ctx, V9fsPath *olddir,

                               const char *old_name, V9fsPath *newdir,

                               const char *new_name)

{

    errno = EPERM;

    return -1;

}
