static void v9fs_readdir(void *opaque)

{

    int32_t fid;

    V9fsFidState *fidp;

    ssize_t retval = 0;

    size_t offset = 7;

    int64_t initial_offset;

    int32_t count, max_count;

    V9fsPDU *pdu = opaque;

    V9fsState *s = pdu->s;



    pdu_unmarshal(pdu, offset, "dqd", &fid, &initial_offset, &max_count);



    trace_v9fs_readdir(pdu->tag, pdu->id, fid, initial_offset, max_count);



    fidp = get_fid(pdu, fid);

    if (fidp == NULL) {

        retval = -EINVAL;

        goto out_nofid;

    }

    if (!fidp->fs.dir) {

        retval = -EINVAL;

        goto out;

    }

    if (initial_offset == 0) {

        v9fs_co_rewinddir(pdu, fidp);

    } else {

        v9fs_co_seekdir(pdu, fidp, initial_offset);

    }

    count = v9fs_do_readdir(pdu, fidp, max_count);

    if (count < 0) {

        retval = count;

        goto out;

    }

    retval = offset;

    retval += pdu_marshal(pdu, offset, "d", count);

    retval += count;

out:

    put_fid(pdu, fidp);

out_nofid:


    complete_pdu(s, pdu, retval);

}