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

    V9fsString fullname;

    V9fsFidState *fidp;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    v9fs_string_init(&fullname);

    pdu_unmarshal(pdu, offset, "dsdddd", &fid, &name, &mode,

                  &major, &minor, &gid);



    fidp = get_fid(s, fid);

    if (fidp == NULL) {

        err = -ENOENT;

        goto out_nofid;

    }

    v9fs_string_sprintf(&fullname, "%s/%s", fidp->path.data, name.data);

    err = v9fs_co_mknod(s, &fullname, fidp->uid, gid,

                        makedev(major, minor), mode);

    if (err < 0) {

        goto out;

    }

    err = v9fs_co_lstat(s, &fullname, &stbuf);

    if (err < 0) {

        goto out;

    }

    stat_to_qid(&stbuf, &qid);

    err = offset;

    err += pdu_marshal(pdu, offset, "Q", &qid);

out:

    put_fid(s, fidp);

out_nofid:

    complete_pdu(s, pdu, err);

    v9fs_string_free(&fullname);

    v9fs_string_free(&name);

}
