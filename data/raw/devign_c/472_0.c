static void cmd646_data_write(void *opaque, target_phys_addr_t addr,

                             uint64_t data, unsigned size)

{

    CMD646BAR *cmd646bar = opaque;



    if (size == 1) {

        ide_ioport_write(cmd646bar->bus, addr, data);

    } else if (addr == 0) {

        if (size == 2) {

            ide_data_writew(cmd646bar->bus, addr, data);

        } else {

            ide_data_writel(cmd646bar->bus, addr, data);

        }

    }

}
