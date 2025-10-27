static void v9fs_open_post_lstat(V9fsState *s, V9fsOpenState *vs, int err)

{

    int flags;



    if (err) {

        err = -errno;

        goto out;

    }



    stat_to_qid(&vs->stbuf, &vs->qid);



    if (S_ISDIR(vs->stbuf.st_mode)) {

        vs->fidp->fs.dir = v9fs_do_opendir(s, &vs->fidp->path);

        v9fs_open_post_opendir(s, vs, err);

    } else {

        if (s->proto_version == V9FS_PROTO_2000L) {

            if (!valid_flags(vs->mode)) {

                err = -EINVAL;

                goto out;

            }

            flags = vs->mode;

        } else {

            flags = omode_to_uflags(vs->mode);

        }

        vs->fidp->fs.fd = v9fs_do_open(s, &vs->fidp->path, flags);

        v9fs_open_post_open(s, vs, err);

    }

    return;

out:

    complete_pdu(s, vs->pdu, err);

    qemu_free(vs);

}
