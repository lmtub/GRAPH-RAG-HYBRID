static void mmio_ide_cmd_write(void *opaque, target_phys_addr_t addr,

                               uint64_t val, unsigned size)

{

    MMIOState *s = opaque;

    ide_cmd_write(&s->bus, 0, val);

}
