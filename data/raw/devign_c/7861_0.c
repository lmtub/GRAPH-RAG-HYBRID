static int local_lremovexattr(FsContext *ctx,

                              const char *path, const char *name)

{

    if ((ctx->fs_sm == SM_MAPPED) &&

        (strncmp(name, "user.virtfs.", 12) == 0)) {

        /*

         * Don't allow fetch of user.virtfs namesapce

         * in case of mapped security

         */

        errno = EACCES;

        return -1;

    }

    return lremovexattr(rpath(ctx, path), name);

}
