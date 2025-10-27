static void dma_mem_writel(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    DMAState *s = opaque;

    uint32_t saddr;



    saddr = (addr & DMA_MASK) >> 2;

    DPRINTF("write dmareg " TARGET_FMT_plx ": 0x%8.8x -> 0x%8.8x\n", addr,

            s->dmaregs[saddr], val);

    switch (saddr) {

    case 0:

        if (val & DMA_INTREN) {

            if (val & DMA_INTR) {

                DPRINTF("Raise IRQ\n");

                qemu_irq_raise(s->irq);

            }

        } else {

            if (s->dmaregs[0] & (DMA_INTR | DMA_INTREN)) {

                DPRINTF("Lower IRQ\n");

                qemu_irq_lower(s->irq);

            }

        }

        if (val & DMA_RESET) {

            qemu_irq_raise(s->dev_reset);

            qemu_irq_lower(s->dev_reset);

        } else if (val & DMA_DRAIN_FIFO) {

            val &= ~DMA_DRAIN_FIFO;

        } else if (val == 0)

            val = DMA_DRAIN_FIFO;

        val &= 0x0fffffff;

        val |= DMA_VER;

        break;

    case 1:

        s->dmaregs[0] |= DMA_LOADED;

        break;

    default:

        break;

    }

    s->dmaregs[saddr] = val;

}
