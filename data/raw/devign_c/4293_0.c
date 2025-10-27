static void iommu_mem_writew(void *opaque, target_phys_addr_t addr,

                             uint32_t val)

{

    IOMMUState *s = opaque;

    target_phys_addr_t saddr;



    saddr = (addr - s->addr) >> 2;

    DPRINTF("write reg[%d] = %x\n", (int)saddr, val);

    switch (saddr) {

    case IOMMU_CTRL:

        switch (val & IOMMU_CTRL_RNGE) {

        case IOMMU_RNGE_16MB:

            s->iostart = 0xffffffffff000000ULL;

            break;

        case IOMMU_RNGE_32MB:

            s->iostart = 0xfffffffffe000000ULL;

            break;

        case IOMMU_RNGE_64MB:

            s->iostart = 0xfffffffffc000000ULL;

            break;

        case IOMMU_RNGE_128MB:

            s->iostart = 0xfffffffff8000000ULL;

            break;

        case IOMMU_RNGE_256MB:

            s->iostart = 0xfffffffff0000000ULL;

            break;

        case IOMMU_RNGE_512MB:

            s->iostart = 0xffffffffe0000000ULL;

            break;

        case IOMMU_RNGE_1GB:

            s->iostart = 0xffffffffc0000000ULL;

            break;

        default:

        case IOMMU_RNGE_2GB:

            s->iostart = 0xffffffff80000000ULL;

            break;

        }

        DPRINTF("iostart = " TARGET_FMT_plx "\n", s->iostart);

        s->regs[saddr] = ((val & IOMMU_CTRL_MASK) | s->version);

        break;

    case IOMMU_BASE:

        s->regs[saddr] = val & IOMMU_BASE_MASK;

        break;

    case IOMMU_TLBFLUSH:

        DPRINTF("tlb flush %x\n", val);

        s->regs[saddr] = val & IOMMU_TLBFLUSH_MASK;

        break;

    case IOMMU_PGFLUSH:

        DPRINTF("page flush %x\n", val);

        s->regs[saddr] = val & IOMMU_PGFLUSH_MASK;

        break;

    case IOMMU_AFAR:

        s->regs[saddr] = val;

        qemu_irq_lower(s->irq);

        break;

    case IOMMU_AFSR:

        s->regs[saddr] = (val & IOMMU_AFSR_MASK) | IOMMU_AFSR_RESV;

        qemu_irq_lower(s->irq);

        break;

    case IOMMU_SBCFG0:

    case IOMMU_SBCFG1:

    case IOMMU_SBCFG2:

    case IOMMU_SBCFG3:

        s->regs[saddr] = val & IOMMU_SBCFG_MASK;

        break;

    case IOMMU_ARBEN:

        // XXX implement SBus probing: fault when reading unmapped

        // addresses, fault cause and address stored to MMU/IOMMU

        s->regs[saddr] = (val & IOMMU_ARBEN_MASK) | IOMMU_MID;

        break;

    default:

        s->regs[saddr] = val;

        break;

    }

}
