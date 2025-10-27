static void v9fs_attach(void *opaque)

{

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;

    int32_t fid, afid, n_uname;

    V9fsString uname, aname;

    V9fsFidState *fidp;

    size_t offset = 7;

    V9fsQID qid;

    ssize_t err;



    pdu_unmarshal(pdu, offset, "ddssd", &fid, &afid, &uname, &aname, &n_uname);

    trace_v9fs_attach(pdu->tag, pdu->id, fid, afid, uname.data, aname.data);



    fidp = alloc_fid(s, fid);

    if (fidp == NULL) {

        err = -EINVAL;

        goto out_nofid;

    }

    fidp->uid = n_uname;

    err = v9fs_co_name_to_path(pdu, NULL, "/", &fidp->path);

    if (err < 0) {

        err = -EINVAL;

        clunk_fid(s, fid);

        goto out;

    }

    err = fid_to_qid(pdu, fidp, &qid);

    if (err < 0) {

        err = -EINVAL;

        clunk_fid(s, fid);

        goto out;

    }

    offset += pdu_marshal(pdu, offset, "Q", &qid);

    err = offset;

out:

    put_fid(pdu, fidp);

out_nofid:



    complete_pdu(s, pdu, err);

    v9fs_string_free(&uname);

    v9fs_string_free(&aname);

}