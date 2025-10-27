static uint64_t qvirtio_scsi_alloc(QVirtIOSCSI *vs, size_t alloc_size,

                                   const void *data)

{

    uint64_t addr;



    addr = guest_alloc(vs->alloc, alloc_size);

    if (data) {

        memwrite(addr, data, alloc_size);

    }



    return addr;

}
