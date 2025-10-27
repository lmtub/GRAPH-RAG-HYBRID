char *desc_get_buf(DescInfo *info, bool read_only)

{

    PCIDevice *dev = PCI_DEVICE(info->ring->r);

    size_t size = read_only ? le16_to_cpu(info->desc.tlv_size) :

                              le16_to_cpu(info->desc.buf_size);



    if (size > info->buf_size) {

        info->buf = g_realloc(info->buf, size);

        info->buf_size = size;

    }



    if (!info->buf) {

        return NULL;

    }



    pci_dma_read(dev, le64_to_cpu(info->desc.buf_addr), info->buf, size);



    return info->buf;

}
