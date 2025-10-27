static void xlnx_dp_set_dpdma(Object *obj, const char *name, Object *val,

                              Error **errp)

{

    XlnxDPState *s = XLNX_DP(obj);

    if (s->console) {

        DisplaySurface *surface = qemu_console_surface(s->console);

        XlnxDPDMAState *dma = XLNX_DPDMA(val);

        xlnx_dpdma_set_host_data_location(dma, DP_GRAPHIC_DMA_CHANNEL,

                                          surface_data(surface));

    }

}
