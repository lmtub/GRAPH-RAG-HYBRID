static ssize_t local_lgetxattr(FsContext *ctx, const char *path,

                               const char *name, void *value, size_t size)

{

    if ((ctx->fs_sm == SM_MAPPED) &&

        (strncmp(name, "user.virtfs.", 12) == 0)) {

        /*

         * Don't allow fetch of user.virtfs namesapce

         * in case of mapped security

         */

        errno = ENOATTR;

        return -1;

    }



    return lgetxattr(rpath(ctx, path), name, value, size);

}
