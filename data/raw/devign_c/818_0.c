static void assigned_dev_ioport_write(void *opaque, target_phys_addr_t addr,

                                      uint64_t data, unsigned size)

{

    assigned_dev_ioport_rw(opaque, addr, size, &data);

}
