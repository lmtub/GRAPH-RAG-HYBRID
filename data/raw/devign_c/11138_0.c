static int proxy_remove(FsContext *ctx, const char *path)

{

    int retval;

    V9fsString name;

    v9fs_string_init(&name);

    v9fs_string_sprintf(&name, "%s", path);

    retval = v9fs_request(ctx->private, T_REMOVE, NULL, "s", &name);

    v9fs_string_free(&name);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
