static XenPTBarFlag xen_pt_bar_reg_parse(XenPCIPassthroughState *s,

                                         XenPTRegInfo *reg)

{

    PCIDevice *d = &s->dev;

    XenPTRegion *region = NULL;

    PCIIORegion *r;

    int index = 0;



    /* check 64bit BAR */

    index = xen_pt_bar_offset_to_index(reg->offset);

    if ((0 < index) && (index < PCI_ROM_SLOT)) {

        int type = s->real_device.io_regions[index - 1].type;



        if ((type & XEN_HOST_PCI_REGION_TYPE_MEM)

            && (type & XEN_HOST_PCI_REGION_TYPE_MEM_64)) {

            region = &s->bases[index - 1];

            if (region->bar_flag != XEN_PT_BAR_FLAG_UPPER) {

                return XEN_PT_BAR_FLAG_UPPER;

            }

        }

    }



    /* check unused BAR */

    r = &d->io_regions[index];

    if (!xen_pt_get_bar_size(r)) {

        return XEN_PT_BAR_FLAG_UNUSED;

    }



    /* for ExpROM BAR */

    if (index == PCI_ROM_SLOT) {

        return XEN_PT_BAR_FLAG_MEM;

    }



    /* check BAR I/O indicator */

    if (s->real_device.io_regions[index].type & XEN_HOST_PCI_REGION_TYPE_IO) {

        return XEN_PT_BAR_FLAG_IO;

    } else {

        return XEN_PT_BAR_FLAG_MEM;

    }

}
