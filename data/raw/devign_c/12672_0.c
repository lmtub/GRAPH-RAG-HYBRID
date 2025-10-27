static int local_lstat(FsContext *fs_ctx, V9fsPath *fs_path, struct stat *stbuf)

{

    int err;

    char buffer[PATH_MAX];

    char *path = fs_path->data;



    err =  lstat(rpath(fs_ctx, path, buffer), stbuf);

    if (err) {

        return err;

    }

    if (fs_ctx->fs_sm == SM_MAPPED) {

        /* Actual credentials are part of extended attrs */

        uid_t tmp_uid;

        gid_t tmp_gid;

        mode_t tmp_mode;

        dev_t tmp_dev;

        if (getxattr(rpath(fs_ctx, path, buffer), "user.virtfs.uid", &tmp_uid,

                    sizeof(uid_t)) > 0) {

            stbuf->st_uid = tmp_uid;

        }

        if (getxattr(rpath(fs_ctx, path, buffer), "user.virtfs.gid", &tmp_gid,

                    sizeof(gid_t)) > 0) {

            stbuf->st_gid = tmp_gid;

        }

        if (getxattr(rpath(fs_ctx, path, buffer), "user.virtfs.mode",

                    &tmp_mode, sizeof(mode_t)) > 0) {

            stbuf->st_mode = tmp_mode;

        }

        if (getxattr(rpath(fs_ctx, path, buffer), "user.virtfs.rdev", &tmp_dev,

                        sizeof(dev_t)) > 0) {

                stbuf->st_rdev = tmp_dev;

        }

    }

    return err;

}
