static void piix3_update_irq_levels(PIIX3State *piix3)

{

    int pirq;



    piix3->pic_levels = 0;

    for (pirq = 0; pirq < PIIX_NUM_PIRQS; pirq++) {

        piix3_set_irq_level(piix3, pirq,

                            pci_bus_get_irq_level(piix3->dev.bus, pirq));

    }

}
