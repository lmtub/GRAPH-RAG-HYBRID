static int local_mkdir(FsContext *fs_ctx, const char *path, FsCred *credp)

{

    int err = -1;

    int serrno = 0;



    /* Determine the security model */

    if (fs_ctx->fs_sm == SM_MAPPED) {

        err = mkdir(rpath(fs_ctx, path), SM_LOCAL_DIR_MODE_BITS);

        if (err == -1) {

            return err;

        }

        credp->fc_mode = credp->fc_mode|S_IFDIR;

        err = local_set_xattr(rpath(fs_ctx, path), credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if (fs_ctx->fs_sm == SM_PASSTHROUGH) {

        err = mkdir(rpath(fs_ctx, path), credp->fc_mode);

        if (err == -1) {

            return err;

        }

        err = local_post_create_passthrough(fs_ctx, path, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    }

    return err;



err_end:

    remove(rpath(fs_ctx, path));

    errno = serrno;

    return err;

}
