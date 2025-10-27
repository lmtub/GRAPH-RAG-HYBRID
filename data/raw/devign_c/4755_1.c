static void v9fs_mknod(void *opaque)

{



    int mode;

    gid_t gid;

    int32_t fid;

    V9fsQID qid;

    int err = 0;

    int major, minor;

    size_t offset = 7;

    V9fsString name;

    struct stat stbuf;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "dsdddd", &fid, &name, &mode,

                  &major, &minor, &gid);




    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    err = v9fs_co_mknod(pdu, fidp, &name, fidp->uid, gid,

                        makedev(major, minor), mode, &stbuf);

    if (err < 0) {

        goto out;

    }

    stat_to_qid(&stbuf, &qid);

    err = offset;

    err += pdu_marshal(pdu, offset, "Q", &qid);

out:

    put_fid(pdu, fidp);

out_nofid:

    trace_v9fs_mknod_return(pdu->tag, pdu->id, qid.type, qid.version, qid.path);

    complete_pdu(s, pdu, err);

    v9fs_string_free(&name);

}