static int local_symlink(FsContext *ctx, const char *oldpath,

                            const char *newpath)

{

    return symlink(oldpath, rpath(ctx, newpath));

}
