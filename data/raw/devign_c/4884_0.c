static void i440fx_realize(PCIDevice *dev, Error **errp)

{

    dev->config[I440FX_SMRAM] = 0x02;



    if (object_property_get_bool(qdev_get_machine(), "iommu", NULL)) {

        error_report("warning: i440fx doesn't support emulated iommu");

    }

}
