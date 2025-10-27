static const char *rpath(FsContext *ctx, const char *path)

{

    /* FIXME: so wrong... */

    static char buffer[4096];

    snprintf(buffer, sizeof(buffer), "%s/%s", ctx->fs_root, path);

    return buffer;

}
