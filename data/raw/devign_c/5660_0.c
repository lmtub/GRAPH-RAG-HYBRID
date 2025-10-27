static void nvic_sysreg_write(void *opaque, target_phys_addr_t addr,

                              uint64_t value, unsigned size)

{

    uint32_t offset = addr;

    if (size == 4) {

        nvic_writel(opaque, offset, value);

        return;

    }

    hw_error("NVIC: Bad write of size %d at offset 0x%x\n", size, offset);

}
