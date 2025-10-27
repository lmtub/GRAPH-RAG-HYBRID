static void vfio_unmap_bars(VFIOPCIDevice *vdev)

{

    int i;



    for (i = 0; i < PCI_ROM_SLOT; i++) {

        vfio_unmap_bar(vdev, i);

    }



    if (vdev->has_vga) {

        vfio_vga_quirk_teardown(vdev);

        pci_unregister_vga(&vdev->pdev);

    }

}
