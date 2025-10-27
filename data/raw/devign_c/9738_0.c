static uint64_t hb_regs_read(void *opaque, target_phys_addr_t offset,

                             unsigned size)

{

    uint32_t *regs = opaque;

    uint32_t value = regs[offset/4];



    if ((offset == 0x100) || (offset == 0x108) || (offset == 0x10C)) {

        value |= 0x30000000;

    }



    return value;

}
