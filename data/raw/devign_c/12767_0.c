static uint64_t assigned_dev_ioport_read(void *opaque,

                                         target_phys_addr_t addr, unsigned size)

{

    return assigned_dev_ioport_rw(opaque, addr, size, NULL);

}
