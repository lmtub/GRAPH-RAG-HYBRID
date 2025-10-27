void stw_tce(VIOsPAPRDevice *dev, uint64_t taddr, uint32_t val)

{

    val = tswap32(val);

    spapr_tce_dma_write(dev, taddr, &val, sizeof(val));

}
