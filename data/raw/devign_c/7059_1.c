static void v9fs_open(void *opaque)

{

    int flags;

    int iounit;

    int32_t fid;

    int32_t mode;

    V9fsQID qid;

    ssize_t err = 0;

    size_t offset = 7;

    struct stat stbuf;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    if (s->proto_version == V9FS_PROTO_2000L) {

        pdu_unmarshal(pdu, offset, "dd", &fid, &mode);

    } else {

        pdu_unmarshal(pdu, offset, "db", &fid, &mode);

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

        offset += pdu_marshal(pdu, offset, "Qd", &qid, 0);

        err = offset;

    } else {

        if (s->proto_version == V9FS_PROTO_2000L) {

            flags = get_dotl_openflags(s, mode);

        } else {

            flags = omode_to_uflags(mode);

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

        offset += pdu_marshal(pdu, offset, "Qd", &qid, iounit);

        err = offset;

    }

out:

    put_fid(pdu, fidp);

out_nofid:



    complete_pdu(s, pdu, err);

}