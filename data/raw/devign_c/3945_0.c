static void ich9_lpc_update_pic(ICH9LPCState *lpc, int gsi)

{

    int i, pic_level;



    assert(gsi < ICH9_LPC_PIC_NUM_PINS);



    /* The pic level is the logical OR of all the PCI irqs mapped to it */

    pic_level = 0;

    for (i = 0; i < ICH9_LPC_NB_PIRQS; i++) {

        int tmp_irq;

        int tmp_dis;

        ich9_lpc_pic_irq(lpc, i, &tmp_irq, &tmp_dis);

        if (!tmp_dis && tmp_irq == gsi) {

            pic_level |= pci_bus_get_irq_level(lpc->d.bus, i);

        }

    }

    if (gsi == lpc->sci_gsi) {

        pic_level |= lpc->sci_level;

    }



    qemu_set_irq(lpc->gsi[gsi], pic_level);

}
