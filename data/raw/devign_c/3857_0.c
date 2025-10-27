static void v9fs_statfs(void *opaque)

{

    int32_t fid;

    ssize_t retval = 0;

    size_t offset = 7;

    V9fsFidState *fidp;

    struct statfs stbuf;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "d", &fid);

    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        retval = -ENOENT;

        goto out_nofid;

    }

    retval = v9fs_co_statfs(pdu, &fidp->path, &stbuf);

    if (retval < 0) {

        goto out;

    }

    retval = offset;

    retval += v9fs_fill_statfs(s, pdu, &stbuf);

out:

    put_fid(pdu, fidp);

out_nofid:

    complete_pdu(s, pdu, retval);

    return;

}
