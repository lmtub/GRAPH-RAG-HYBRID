void aio_bh_call(QEMUBH *bh)

{

    bh->cb(bh->opaque);

}
