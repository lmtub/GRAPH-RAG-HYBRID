static void megasas_port_write(void *opaque, target_phys_addr_t addr,

                               uint64_t val, unsigned size)

{

    megasas_mmio_write(opaque, addr & 0xff, val, size);

}
