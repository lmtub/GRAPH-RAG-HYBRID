static int vfio_pci_hot_reset_multi(VFIOPCIDevice *vdev)

{

    return vfio_pci_hot_reset(vdev, false);

}
