static int local_lstat(FsContext *fs_ctx, V9fsPath *fs_path, struct stat *stbuf)

{

    int err;

    char *buffer;

    char *path = fs_path->data;



    buffer = rpath(fs_ctx, path);

    err =  lstat(buffer, stbuf);

    if (err) {

        goto err_out;

    }

    if (fs_ctx->export_flags & V9FS_SM_MAPPED) {

        /* Actual credentials are part of extended attrs */

        uid_t tmp_uid;

        gid_t tmp_gid;

        mode_t tmp_mode;

        dev_t tmp_dev;

        if (getxattr(buffer, "user.virtfs.uid", &tmp_uid, sizeof(uid_t)) > 0) {

            stbuf->st_uid = le32_to_cpu(tmp_uid);

        }

        if (getxattr(buffer, "user.virtfs.gid", &tmp_gid, sizeof(gid_t)) > 0) {

            stbuf->st_gid = le32_to_cpu(tmp_gid);

        }

        if (getxattr(buffer, "user.virtfs.mode",

                    &tmp_mode, sizeof(mode_t)) > 0) {

            stbuf->st_mode = le32_to_cpu(tmp_mode);

        }

        if (getxattr(buffer, "user.virtfs.rdev", &tmp_dev, sizeof(dev_t)) > 0) {

            stbuf->st_rdev = le64_to_cpu(tmp_dev);

        }

    } else if (fs_ctx->export_flags & V9FS_SM_MAPPED_FILE) {

        local_mapped_file_attr(fs_ctx, path, stbuf);

    }



err_out:

    g_free(buffer);

    return err;

}
