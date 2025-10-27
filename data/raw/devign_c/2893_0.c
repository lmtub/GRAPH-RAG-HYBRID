static uint32_t iommu_mem_readw(void *opaque, target_phys_addr_t addr)

{

    IOMMUState *s = opaque;

    target_phys_addr_t saddr;

    uint32_t ret;



    saddr = (addr - s->addr) >> 2;

    switch (saddr) {

    default:

        ret = s->regs[saddr];

        break;

    case IOMMU_AFAR:

    case IOMMU_AFSR:

        ret = s->regs[saddr];

        qemu_irq_lower(s->irq);

        break;

    }

    DPRINTF("read reg[%d] = %x\n", (int)saddr, ret);

    return ret;

}
