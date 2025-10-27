static ssize_t local_readlink(FsContext *ctx, const char *path,

                                char *buf, size_t bufsz)

{

    return readlink(rpath(ctx, path), buf, bufsz);

}
