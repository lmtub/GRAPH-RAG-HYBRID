vmxnet3_pop_rxc_descr(VMXNET3State *s, int qidx, uint32_t *descr_gen)

{

    uint8_t ring_gen;

    struct Vmxnet3_RxCompDesc rxcd;



    hwaddr daddr =

        vmxnet3_ring_curr_cell_pa(&s->rxq_descr[qidx].comp_ring);



    pci_dma_read(PCI_DEVICE(s), daddr,

                 &rxcd, sizeof(struct Vmxnet3_RxCompDesc));



    ring_gen = vmxnet3_ring_curr_gen(&s->rxq_descr[qidx].comp_ring);



    if (rxcd.gen != ring_gen) {

        *descr_gen = ring_gen;

        vmxnet3_inc_rx_completion_counter(s, qidx);

        return daddr;

    }



    return 0;

}
