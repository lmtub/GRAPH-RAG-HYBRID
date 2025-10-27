QEMUBH *qemu_bh_new(QEMUBHFunc *cb, void *opaque)

{

    QEMUBH *bh;



    bh = qemu_malloc(sizeof(*bh));

    bh->cb = cb;

    bh->opaque = opaque;



    return bh;

}
