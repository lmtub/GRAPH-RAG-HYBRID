static int local_fstat(FsContext *fs_ctx, int fid_type,

                       V9fsFidOpenState *fs, struct stat *stbuf)

{

    int err, fd;



    if (fid_type == P9_FID_DIR) {

        fd = dirfd(fs->dir);

    } else {

        fd = fs->fd;

    }



    err = fstat(fd, stbuf);

    if (err) {

        return err;

    }

    if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        /* Actual credentials are part of extended attrs */

        uid_t tmp_uid;

        gid_t tmp_gid;

        mode_t tmp_mode;

        dev_t tmp_dev;



        if (fgetxattr(fd, "user.virtfs.uid",

                      &tmp_uid, sizeof(uid_t)) > 0) {

            stbuf->st_uid = tmp_uid;

        }

        if (fgetxattr(fd, "user.virtfs.gid",

                      &tmp_gid, sizeof(gid_t)) > 0) {

            stbuf->st_gid = tmp_gid;

        }

        if (fgetxattr(fd, "user.virtfs.mode",

                      &tmp_mode, sizeof(mode_t)) > 0) {

            stbuf->st_mode = tmp_mode;

        }

        if (fgetxattr(fd, "user.virtfs.rdev",

                      &tmp_dev, sizeof(dev_t)) > 0) {

                stbuf->st_rdev = tmp_dev;

        }

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        errno = EOPNOTSUPP;

        return -1;

    }

    return err;

}
