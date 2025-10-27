static void coroutine_fn v9fs_open(void *opaque)

{

    int flags;

    int32_t fid;

    int32_t mode;

    V9fsQID qid;

    int iounit = 0;

    ssize_t err = 0;

    size_t offset = 7;

    struct stat stbuf;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    if (s->proto_version == V9FS_PROTO_2000L) {

        err = pdu_unmarshal(pdu, offset, "dd", &fid, &mode);

    } else {

        uint8_t modebyte;

        err = pdu_unmarshal(pdu, offset, "db", &fid, &modebyte);

        mode = modebyte;

    }

    if (err < 0) {

        goto out_nofid;

    }

    trace_v9fs_open(pdu->tag, pdu->id, fid, mode);



    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    BUG_ON(fidp->fid_type != P9_FID_NONE);



    err = v9fs_co_lstat(pdu, &fidp->path, &stbuf);

    if (err < 0) {

        goto out;

    }

    stat_to_qid(&stbuf, &qid);

    if (S_ISDIR(stbuf.st_mode)) {

        err = v9fs_co_opendir(pdu, fidp);

        if (err < 0) {

            goto out;

        }

        fidp->fid_type = P9_FID_DIR;

        err = pdu_marshal(pdu, offset, "Qd", &qid, 0);

        if (err < 0) {

            goto out;

        }

        err += offset;

    } else {

        if (s->proto_version == V9FS_PROTO_2000L) {

            flags = get_dotl_openflags(s, mode);

        } else {

            flags = omode_to_uflags(mode);

        }

        if (is_ro_export(&s->ctx)) {

            if (mode & O_WRONLY || mode & O_RDWR ||

                mode & O_APPEND || mode & O_TRUNC) {

                err = -EROFS;

                goto out;

            }

        }

        err = v9fs_co_open(pdu, fidp, flags);

        if (err < 0) {

            goto out;

        }

        fidp->fid_type = P9_FID_FILE;

        fidp->open_flags = flags;

        if (flags & O_EXCL) {

            /*

             * We let the host file system do O_EXCL check

             * We should not reclaim such fd

             */

            fidp->flags |= FID_NON_RECLAIMABLE;

        }

        iounit = get_iounit(pdu, &fidp->path);

        err = pdu_marshal(pdu, offset, "Qd", &qid, iounit);

        if (err < 0) {

            goto out;

        }

        err += offset;

    }

    trace_v9fs_open_return(pdu->tag, pdu->id,

                           qid.type, qid.version, qid.path, iounit);

out:

    put_fid(pdu, fidp);

out_nofid:

    pdu_complete(pdu, err);

}
