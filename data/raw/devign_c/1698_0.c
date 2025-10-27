static void mptsas_scsi_uninit(PCIDevice *dev)

{

    MPTSASState *s = MPT_SAS(dev);



    qemu_bh_delete(s->request_bh);

    if (s->msi_in_use) {

        msi_uninit(dev);

    }

}
