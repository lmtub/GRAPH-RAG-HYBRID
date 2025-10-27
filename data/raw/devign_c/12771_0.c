static void ne2000_write(void *opaque, target_phys_addr_t addr,

                         uint64_t data, unsigned size)

{

    NE2000State *s = opaque;



    if (addr < 0x10 && size == 1) {

        ne2000_ioport_write(s, addr, data);

    } else if (addr == 0x10) {

        if (size <= 2) {

            ne2000_asic_ioport_write(s, addr, data);

        } else {

            ne2000_asic_ioport_writel(s, addr, data);

        }

    } else if (addr == 0x1f && size == 1) {

        ne2000_reset_ioport_write(s, addr, data);

    }

}
