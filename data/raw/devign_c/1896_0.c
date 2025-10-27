static int proxy_symlink(FsContext *fs_ctx, const char *oldpath,

                         V9fsPath *dir_path, const char *name, FsCred *credp)

{

    int retval;

    V9fsString fullname, target;



    v9fs_string_init(&fullname);

    v9fs_string_init(&target);



    v9fs_string_sprintf(&fullname, "%s/%s", dir_path->data, name);

    v9fs_string_sprintf(&target, "%s", oldpath);



    retval = v9fs_request(fs_ctx->private, T_SYMLINK, NULL, "ssdd",

                          &target, &fullname, credp->fc_uid, credp->fc_gid);

    v9fs_string_free(&fullname);

    v9fs_string_free(&target);

    if (retval < 0) {

        errno = -retval;

        retval = -1;

    }

    return retval;

}
