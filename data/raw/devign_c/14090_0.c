static void v9fs_link(void *opaque)

{

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;

    int32_t dfid, oldfid;

    V9fsFidState *dfidp, *oldfidp;

    V9fsString name;

    size_t offset = 7;

    int err = 0;



    pdu_unmarshal(pdu, offset, "dds", &dfid, &oldfid, &name);

    trace_v9fs_link(pdu->tag, pdu->id, dfid, oldfid, name.data);



    dfidp = get_fid(pdu, dfid);

    if (dfidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }



    oldfidp = get_fid(pdu, oldfid);

    if (oldfidp == NULL) {

        err = -ENOENT;

        goto out;

    }

    err = v9fs_co_link(pdu, oldfidp, dfidp, &name);

    if (!err) {

        err = offset;

    }

out:

    put_fid(pdu, dfidp);

out_nofid:

    v9fs_string_free(&name);

    complete_pdu(s, pdu, err);

}
