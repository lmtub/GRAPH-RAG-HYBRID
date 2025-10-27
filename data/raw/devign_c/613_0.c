static int coroutine_fn v9fs_do_readdir_with_stat(V9fsPDU *pdu,

                                                  V9fsFidState *fidp,

                                                  uint32_t max_count)

{

    V9fsPath path;

    V9fsStat v9stat;

    int len, err = 0;

    int32_t count = 0;

    struct stat stbuf;

    off_t saved_dir_pos;

    struct dirent *dent;



    /* save the directory position */

    saved_dir_pos = v9fs_co_telldir(pdu, fidp);

    if (saved_dir_pos < 0) {

        return saved_dir_pos;

    }



    while (1) {

        v9fs_path_init(&path);



        v9fs_readdir_lock(&fidp->fs.dir);



        err = v9fs_co_readdir(pdu, fidp, &dent);

        if (err || !dent) {

            break;

        }

        err = v9fs_co_name_to_path(pdu, &fidp->path, dent->d_name, &path);

        if (err < 0) {

            break;

        }

        err = v9fs_co_lstat(pdu, &path, &stbuf);

        if (err < 0) {

            break;

        }

        err = stat_to_v9stat(pdu, &path, dent->d_name, &stbuf, &v9stat);

        if (err < 0) {

            break;

        }

        /* 11 = 7 + 4 (7 = start offset, 4 = space for storing count) */

        len = pdu_marshal(pdu, 11 + count, "S", &v9stat);



        v9fs_readdir_unlock(&fidp->fs.dir);



        if ((len != (v9stat.size + 2)) || ((count + len) > max_count)) {

            /* Ran out of buffer. Set dir back to old position and return */

            v9fs_co_seekdir(pdu, fidp, saved_dir_pos);

            v9fs_stat_free(&v9stat);

            v9fs_path_free(&path);

            return count;

        }

        count += len;

        v9fs_stat_free(&v9stat);

        v9fs_path_free(&path);

        saved_dir_pos = dent->d_off;

    }



    v9fs_readdir_unlock(&fidp->fs.dir);



    v9fs_path_free(&path);

    if (err < 0) {

        return err;

    }

    return count;

}
