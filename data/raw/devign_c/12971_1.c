static void qemu_gluster_complete_aio(void *opaque)

{

    GlusterAIOCB *acb = (GlusterAIOCB *)opaque;



    qemu_bh_delete(acb->bh);

    acb->bh = NULL;

    qemu_coroutine_enter(acb->coroutine, NULL);

}
