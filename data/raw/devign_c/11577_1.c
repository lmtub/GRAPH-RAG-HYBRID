static void v9fs_stat(void *opaque)

{

    int32_t fid;

    V9fsStat v9stat;

    ssize_t err = 0;

    size_t offset = 7;

    struct stat stbuf;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "d", &fid);




    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    err = v9fs_co_lstat(pdu, &fidp->path, &stbuf);

    if (err < 0) {

        goto out;

    }

    err = stat_to_v9stat(pdu, &fidp->path, &stbuf, &v9stat);

    if (err < 0) {

        goto out;

    }

    offset += pdu_marshal(pdu, offset, "wS", 0, &v9stat);

    err = offset;

    v9fs_stat_free(&v9stat);

out:

    put_fid(pdu, fidp);

out_nofid:

    trace_v9fs_stat_return(pdu->tag, pdu->id, v9stat.mode,

                           v9stat.atime, v9stat.mtime, v9stat.length);



    complete_pdu(s, pdu, err);

}