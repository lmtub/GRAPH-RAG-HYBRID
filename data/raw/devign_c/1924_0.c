static int do_create_others(int type, struct iovec *iovec)

{

    dev_t rdev;

    int retval = 0;

    int offset = PROXY_HDR_SZ;

    V9fsString oldpath, path;

    int mode, uid, gid, cur_uid, cur_gid;



    v9fs_string_init(&path);

    v9fs_string_init(&oldpath);

    cur_uid = geteuid();

    cur_gid = getegid();



    retval = proxy_unmarshal(iovec, offset, "dd", &uid, &gid);

    if (retval < 0) {

        return retval;

    }

    offset += retval;

    retval = setfsugid(uid, gid);

    if (retval < 0) {

        retval = -errno;

        goto err_out;

    }

    switch (type) {

    case T_MKNOD:

        retval = proxy_unmarshal(iovec, offset, "sdq", &path, &mode, &rdev);

        if (retval < 0) {

            goto err_out;

        }

        retval = mknod(path.data, mode, rdev);

        break;

    case T_MKDIR:

        retval = proxy_unmarshal(iovec, offset, "sd", &path, &mode);

        if (retval < 0) {

            goto err_out;

        }

        retval = mkdir(path.data, mode);

        break;

    case T_SYMLINK:

        retval = proxy_unmarshal(iovec, offset, "ss", &oldpath, &path);

        if (retval < 0) {

            goto err_out;

        }

        retval = symlink(oldpath.data, path.data);

        break;

    }

    if (retval < 0) {

        retval = -errno;

    }



err_out:

    v9fs_string_free(&path);

    v9fs_string_free(&oldpath);

    setfsugid(cur_uid, cur_gid);

    return retval;

}
