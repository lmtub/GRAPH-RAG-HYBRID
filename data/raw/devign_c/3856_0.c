static void ehci_flush_qh(EHCIQueue *q)

{

    uint32_t *qh = (uint32_t *) &q->qh;

    uint32_t dwords = sizeof(EHCIqh) >> 2;

    uint32_t addr = NLPTR_GET(q->qhaddr);



    put_dwords(addr + 3 * sizeof(uint32_t), qh + 3, dwords - 3);

}
