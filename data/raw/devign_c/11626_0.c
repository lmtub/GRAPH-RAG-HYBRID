static int local_open2(FsContext *ctx, const char *path, int flags, mode_t mode)

{

    return open(rpath(ctx, path), flags, mode);

}
