static uint32_t esp_mem_readb(void *opaque, target_phys_addr_t addr)

{

    ESPState *s = opaque;

    uint32_t saddr;



    saddr = (addr >> s->it_shift) & (ESP_REGS - 1);

    DPRINTF("read reg[%d]: 0x%2.2x\n", saddr, s->rregs[saddr]);

    switch (saddr) {

    case ESP_FIFO:

        if (s->ti_size > 0) {

            s->ti_size--;

            if ((s->rregs[ESP_RSTAT] & STAT_PIO_MASK) == 0) {

                /* Data in/out.  */

                fprintf(stderr, "esp: PIO data read not implemented\n");

                s->rregs[ESP_FIFO] = 0;

            } else {

                s->rregs[ESP_FIFO] = s->ti_buf[s->ti_rptr++];

            }

            esp_raise_irq(s);

        }

        if (s->ti_size == 0) {

            s->ti_rptr = 0;

            s->ti_wptr = 0;

        }

        break;

    case ESP_RINTR:

        // Clear interrupt/error status bits

        s->rregs[ESP_RSTAT] &= ~(STAT_GE | STAT_PE);

        esp_lower_irq(s);

        break;

    default:

        break;

    }

    return s->rregs[saddr];

}
