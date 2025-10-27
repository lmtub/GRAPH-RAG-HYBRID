static void piix3_set_irq(void *opaque, int irq_num, int level)

{

    int i, pic_irq, pic_level;

    PIIX3State *piix3 = opaque;



    /* now we change the pic irq level according to the piix irq mappings */

    /* XXX: optimize */

    pic_irq = piix3->dev.config[0x60 + irq_num];

    if (pic_irq < 16) {

        /* The pic level is the logical OR of all the PCI irqs mapped

           to it */

        pic_level = 0;

        for (i = 0; i < 4; i++) {

            if (pic_irq == piix3->dev.config[0x60 + i]) {

                pic_level |= pci_bus_get_irq_level(piix3->dev.bus, i);

            }

        }

        qemu_set_irq(piix3->pic[pic_irq], pic_level);

    }

}
