vmxnet3_pop_next_tx_descr(VMXNET3State *s,

                          int qidx,

                          struct Vmxnet3_TxDesc *txd,

                          uint32_t *descr_idx)

{

    Vmxnet3Ring *ring = &s->txq_descr[qidx].tx_ring;

    PCIDevice *d = PCI_DEVICE(s);



    vmxnet3_ring_read_curr_cell(d, ring, txd);

    if (txd->gen == vmxnet3_ring_curr_gen(ring)) {

        /* Only read after generation field verification */

        smp_rmb();

        /* Re-read to be sure we got the latest version */

        vmxnet3_ring_read_curr_cell(d, ring, txd);

        VMXNET3_RING_DUMP(VMW_RIPRN, "TX", qidx, ring);

        *descr_idx = vmxnet3_ring_curr_cell_idx(ring);

        vmxnet3_inc_tx_consumption_counter(s, qidx);

        return true;

    }



    return false;

}
