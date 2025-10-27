static int local_symlink(FsContext *fs_ctx, const char *oldpath,

        const char *newpath, FsCred *credp)

{

    int err = -1;

    int serrno = 0;



    /* Determine the security model */

    if (fs_ctx->fs_sm == SM_MAPPED) {

        int fd;

        ssize_t oldpath_size, write_size;

        fd = open(rpath(fs_ctx, newpath), O_CREAT|O_EXCL|O_RDWR,

                SM_LOCAL_MODE_BITS);

        if (fd == -1) {

            return fd;

        }

        /* Write the oldpath (target) to the file. */

        oldpath_size = strlen(oldpath) + 1;

        do {

            write_size = write(fd, (void *)oldpath, oldpath_size);

        } while (write_size == -1 && errno == EINTR);



        if (write_size != oldpath_size) {

            serrno = errno;

            close(fd);

            err = -1;

            goto err_end;

        }

        close(fd);

        /* Set cleint credentials in symlink's xattr */

        credp->fc_mode = credp->fc_mode|S_IFLNK;

        err = local_set_xattr(rpath(fs_ctx, newpath), credp);

        if (err == -1) {

            serrno = errno;

            goto err_end;

        }

    } else if ((fs_ctx->fs_sm == SM_PASSTHROUGH) ||

               (fs_ctx->fs_sm == SM_NONE)) {

        err = symlink(oldpath, rpath(fs_ctx, newpath));

        if (err) {

            return err;

        }

        err = lchown(rpath(fs_ctx, newpath), credp->fc_uid, credp->fc_gid);

        if (err == -1) {

            /*

             * If we fail to change ownership and if we are

             * using security model none. Ignore the error

             */

            if (fs_ctx->fs_sm != SM_NONE) {

                serrno = errno;

                goto err_end;

            } else

                err = 0;

        }

    }

    return err;



err_end:

    remove(rpath(fs_ctx, newpath));

    errno = serrno;

    return err;

}
