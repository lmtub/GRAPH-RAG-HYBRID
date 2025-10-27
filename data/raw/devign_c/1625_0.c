static void rtce_init(VIOsPAPRDevice *dev)

{

    size_t size = (dev->rtce_window_size >> SPAPR_VIO_TCE_PAGE_SHIFT)

        * sizeof(VIOsPAPR_RTCE);



    if (size) {

        dev->rtce_table = g_malloc0(size);

    }

}
