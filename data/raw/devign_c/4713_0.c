static uint64_t cmd646_data_read(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    CMD646BAR *cmd646bar = opaque;



    if (size == 1) {

        return ide_ioport_read(cmd646bar->bus, addr);

    } else if (addr == 0) {

        if (size == 2) {

            return ide_data_readw(cmd646bar->bus, addr);

        } else {

            return ide_data_readl(cmd646bar->bus, addr);

        }

    }

    return ((uint64_t)1 << (size * 8)) - 1;

}
