static int local_open2(FsContext *fs_ctx, const char *path, int flags,

        FsCred *credp)

{

    int fd = -1;

    int err = -1;

    int serrno = 0;



    /* Determine the security model */

    if (fs_ctx->fs_sm == SM_MAPPED) {

        fd = open(rpath(fs_ctx, path), flags, SM_LOCAL_MODE_BITS);

        if (fd == -1) {

            return fd;

        }

        credp->fc_mode = credp->fc_mode|S_IFREG;

        /* Set cleint credentials in xattr */

        err = local_set_xattr(rpath(fs_ctx, path), credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if (fs_ctx->fs_sm == SM_PASSTHROUGH) {

        fd = open(rpath(fs_ctx, path), flags, credp->fc_mode);

        if (fd == -1) {

            return fd;

        }

        err = local_post_create_passthrough(fs_ctx, path, credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    }

    return fd;



err_end:

    close(fd);

    remove(rpath(fs_ctx, path));

    errno = serrno;

    return err;

}
