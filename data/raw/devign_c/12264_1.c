static int local_chown(FsContext *fs_ctx, const char *path, FsCred *credp)

{

    if ((credp->fc_uid == -1 && credp->fc_gid == -1) ||

            (fs_ctx->fs_sm == SM_PASSTHROUGH)) {

        return lchown(rpath(fs_ctx, path), credp->fc_uid, credp->fc_gid);

    } else if (fs_ctx->fs_sm == SM_MAPPED) {

        return local_set_xattr(rpath(fs_ctx, path), credp);

    } else if (fs_ctx->fs_sm == SM_PASSTHROUGH) {

        return lchown(rpath(fs_ctx, path), credp->fc_uid, credp->fc_gid);

    }

    return -1;

}
