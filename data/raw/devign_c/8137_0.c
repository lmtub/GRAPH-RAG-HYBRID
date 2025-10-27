static void q35_host_get_pci_hole64_start(Object *obj, Visitor *v,

                                          const char *name, void *opaque,

                                          Error **errp)

{

    PCIHostState *h = PCI_HOST_BRIDGE(obj);

    Range w64;



    pci_bus_get_w64_range(h->bus, &w64);



    visit_type_uint64(v, name, &w64.begin, errp);

}
