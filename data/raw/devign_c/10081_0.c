static uint32_t gt64120_read_config(PCIDevice *d, uint32_t address, int len)

{

    uint32_t val = pci_default_read_config(d, address, len);

#ifdef TARGET_WORDS_BIGENDIAN

    val = bswap32(val);

#endif

    return val;

}
