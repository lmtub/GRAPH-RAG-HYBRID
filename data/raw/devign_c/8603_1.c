static int qemu_rdma_registration_start(QEMUFile *f, void *opaque,

                                        uint64_t flags)

{

    QEMUFileRDMA *rfile = opaque;

    RDMAContext *rdma = rfile->rdma;



    CHECK_ERROR_STATE();



    DDDPRINTF("start section: %" PRIu64 "\n", flags);

    qemu_put_be64(f, RAM_SAVE_FLAG_HOOK);

    qemu_fflush(f);



    return 0;

}
