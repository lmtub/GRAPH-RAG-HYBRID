static void spapr_vio_quiesce_one(VIOsPAPRDevice *dev)

{

    dev->flags &= ~VIO_PAPR_FLAG_DMA_BYPASS;



    if (dev->rtce_table) {

        size_t size = (dev->rtce_window_size >> SPAPR_VIO_TCE_PAGE_SHIFT)

            * sizeof(VIOsPAPR_RTCE);

        memset(dev->rtce_table, 0, size);

    }



    dev->crq.qladdr = 0;

    dev->crq.qsize = 0;

    dev->crq.qnext = 0;

}
