pvscsi_cleanup_msi(PVSCSIState *s)

{

    PCIDevice *d = PCI_DEVICE(s);



    if (s->msi_used) {

        msi_uninit(d);

    }

}
