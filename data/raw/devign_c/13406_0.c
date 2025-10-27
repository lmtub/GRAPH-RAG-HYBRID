static void i440fx_pcihost_get_pci_hole_end(Object *obj, Visitor *v,

                                            const char *name, void *opaque,

                                            Error **errp)

{

    I440FXState *s = I440FX_PCI_HOST_BRIDGE(obj);

    uint32_t value = s->pci_hole.end;



    visit_type_uint32(v, name, &value, errp);

}
