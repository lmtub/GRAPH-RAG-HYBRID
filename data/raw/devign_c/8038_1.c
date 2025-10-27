void stb_tce(VIOsPAPRDevice *dev, uint64_t taddr, uint8_t val)

{

    spapr_tce_dma_write(dev, taddr, &val, sizeof(val));

}
