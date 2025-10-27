static int intel_hda_exit(PCIDevice *pci)

{

    IntelHDAState *d = DO_UPCAST(IntelHDAState, pci, pci);



    if (d->msi) {

        msi_uninit(&d->pci);

    }

    cpu_unregister_io_memory(d->mmio_addr);

    return 0;

}
