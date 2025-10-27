static void v9fs_clunk(void *opaque)

{

    int err;

    int32_t fid;

    size_t offset = 7;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "d", &fid);




    fidp = clunk_fid(s, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    /*

     * Bump the ref so that put_fid will

     * free the fid.

     */

    fidp->ref++;

    err = offset;



    put_fid(pdu, fidp);

out_nofid:

    complete_pdu(s, pdu, err);

}