static int local_chown(FsContext *fs_ctx, V9fsPath *fs_path, FsCred *credp)

{

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    if ((credp->fc_uid == -1 && credp->fc_gid == -1) ||

        (fs_ctx->export_flags & V9FS_SM_PASSTHROUGH) ||

        (fs_ctx->export_flags & V9FS_SM_NONE)) {

        return lchown(rpath(fs_ctx, path, buffer),

                      credp->fc_uid, credp->fc_gid);

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        return local_set_xattr(rpath(fs_ctx, path, buffer), credp);

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        return local_set_mapped_file_attr(fs_ctx, path, credp);

    }

    return -1;

}
