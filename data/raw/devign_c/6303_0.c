static bool virtio_pci_modern_state_needed(void *opaque)

{

    VirtIOPCIProxy *proxy = opaque;



    return !(proxy->flags & VIRTIO_PCI_FLAG_DISABLE_MODERN);

}
