static int v9fs_synth_rename(FsContext *ctx, const char *oldpath,

                             const char *newpath)

{

    errno = EPERM;

    return -1;

}
