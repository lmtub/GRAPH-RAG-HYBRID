static void v9fs_lcreate(void *opaque)

{

    int32_t dfid, flags, mode;

    gid_t gid;

    ssize_t err = 0;

    ssize_t offset = 7;

    V9fsString name;

    V9fsFidState *fidp;

    struct stat stbuf;

    V9fsQID qid;

    int32_t iounit;

    V9fsPDU *pdu = opaque;



    pdu_unmarshal(pdu, offset, "dsddd", &dfid, &name, &flags,

                  &mode, &gid);




    fidp = get_fid(pdu, dfid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }



    flags = get_dotl_openflags(pdu->s, flags);

    err = v9fs_co_open2(pdu, fidp, &name, gid,

                        flags | O_CREAT, mode, &stbuf);

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

    iounit =  get_iounit(pdu, &fidp->path);

    stat_to_qid(&stbuf, &qid);

    offset += pdu_marshal(pdu, offset, "Qd", &qid, iounit);

    err = offset;

out:

    put_fid(pdu, fidp);

out_nofid:

    trace_v9fs_lcreate_return(pdu->tag, pdu->id,

                              qid.type, qid.version, qid.path, iounit);

    complete_pdu(pdu->s, pdu, err);

    v9fs_string_free(&name);

}