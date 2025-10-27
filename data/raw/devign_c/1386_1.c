static void v9fs_readlink(void *opaque)

{

    V9fsPDU *pdu = opaque;

    size_t offset = 7;

    V9fsString target;

    int32_t fid;

    int err = 0;

    V9fsFidState *fidp;



    pdu_unmarshal(pdu, offset, "d", &fid);


    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }



    v9fs_string_init(&target);

    err = v9fs_co_readlink(pdu, &fidp->path, &target);

    if (err < 0) {

        goto out;

    }

    offset += pdu_marshal(pdu, offset, "s", &target);

    err = offset;

    v9fs_string_free(&target);

out:

    put_fid(pdu, fidp);

out_nofid:

    trace_v9fs_readlink_return(pdu->tag, pdu->id, target.data);

    complete_pdu(pdu->s, pdu, err);

}