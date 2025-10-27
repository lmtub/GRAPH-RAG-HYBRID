static int proxy_chown(FsContext *fs_ctx, V9fsPath *fs_path, FsCred *credp)

{

    int retval;

    retval = v9fs_request(fs_ctx->private, T_CHOWN, NULL, "sdd",

                          fs_path, credp->fc_uid, credp->fc_gid);

    if (retval < 0) {

        errno = -retval;

    }

    return retval;

}
