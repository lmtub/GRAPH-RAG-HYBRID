static uint8_t ide_wait_clear(uint8_t flag)

{

    QPCIDevice *dev;

    QPCIBar bmdma_bar, ide_bar;

    uint8_t data;

    time_t st;



    dev = get_pci_device(&bmdma_bar, &ide_bar);



    /* Wait with a 5 second timeout */

    time(&st);

    while (true) {

        data = qpci_io_readb(dev, ide_bar, reg_status);

        if (!(data & flag)) {


            return data;

        }

        if (difftime(time(NULL), st) > 5.0) {

            break;

        }

        nsleep(400);

    }

    g_assert_not_reached();

}