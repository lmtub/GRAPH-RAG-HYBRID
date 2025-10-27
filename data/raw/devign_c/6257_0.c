static void omap_sti_fifo_write(void *opaque, target_phys_addr_t addr,

                                uint64_t value, unsigned size)

{

    struct omap_sti_s *s = (struct omap_sti_s *) opaque;

    int ch = addr >> 6;

    uint8_t byte = value;



    if (size != 1) {

        return omap_badwidth_write8(opaque, addr, size);

    }



    if (ch == STI_TRACE_CONTROL_CHANNEL) {

        /* Flush channel <i>value</i>.  */

        qemu_chr_fe_write(s->chr, (const uint8_t *) "\r", 1);

    } else if (ch == STI_TRACE_CONSOLE_CHANNEL || 1) {

        if (value == 0xc0 || value == 0xc3) {

            /* Open channel <i>ch</i>.  */

        } else if (value == 0x00)

            qemu_chr_fe_write(s->chr, (const uint8_t *) "\n", 1);

        else

            qemu_chr_fe_write(s->chr, &byte, 1);

    }

}
