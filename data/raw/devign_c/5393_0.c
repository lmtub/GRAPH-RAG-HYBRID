static uint64_t iack_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    return pic_read_irq(isa_pic);

}
