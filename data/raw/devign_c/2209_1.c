static int proxy_mkdir(FsContext *fs_ctx, V9fsPath *dir_path,

                       const char *name, FsCred *credp)

{

    int retval;

    V9fsString fullname;



    v9fs_string_init(&fullname);

    v9fs_string_sprintf(&fullname, "%s/%s", dir_path->data, name);



    retval = v9fs_request(fs_ctx->private, T_MKDIR, NULL, &fullname,

                          credp->fc_mode, credp->fc_uid, credp->fc_gid);

    v9fs_string_free(&fullname);

    if (retval < 0) {

        errno = -retval;

        retval = -1;

    }

    v9fs_string_free(&fullname);

    return retval;

}
