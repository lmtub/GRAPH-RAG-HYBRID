static void vmxnet3_fill_stats(VMXNET3State *s)

{

    int i;



    if (!s->device_active)

        return;



    for (i = 0; i < s->txq_num; i++) {

        pci_dma_write(PCI_DEVICE(s),

                      s->txq_descr[i].tx_stats_pa,

                      &s->txq_descr[i].txq_stats,

                      sizeof(s->txq_descr[i].txq_stats));

    }



    for (i = 0; i < s->rxq_num; i++) {

        pci_dma_write(PCI_DEVICE(s),

                      s->rxq_descr[i].rx_stats_pa,

                      &s->rxq_descr[i].rxq_stats,

                      sizeof(s->rxq_descr[i].rxq_stats));

    }

}
