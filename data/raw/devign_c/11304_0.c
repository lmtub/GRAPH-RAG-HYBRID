void shpc_cleanup(PCIDevice *d, MemoryRegion *bar)

{

    SHPCDevice *shpc = d->shpc;

    d->cap_present &= ~QEMU_PCI_CAP_SHPC;

    memory_region_del_subregion(bar, &shpc->mmio);

    object_unparent(OBJECT(&shpc->mmio));

    /* TODO: cleanup config space changes? */

    g_free(shpc->config);

    g_free(shpc->cmask);

    g_free(shpc->wmask);

    g_free(shpc->w1cmask);

    g_free(shpc);

}
