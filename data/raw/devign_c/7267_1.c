static void pc_dimm_get_size(Object *obj, Visitor *v, const char *name,

                             void *opaque, Error **errp)

{

    uint64_t value;

    MemoryRegion *mr;

    PCDIMMDevice *dimm = PC_DIMM(obj);

    PCDIMMDeviceClass *ddc = PC_DIMM_GET_CLASS(obj);



    mr = ddc->get_memory_region(dimm);

    value = memory_region_size(mr);



    visit_type_uint64(v, name, &value, errp);

}
