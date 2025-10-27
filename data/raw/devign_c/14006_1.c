static void nvdimm_realize(PCDIMMDevice *dimm, Error **errp)

{

    MemoryRegion *mr = host_memory_backend_get_memory(dimm->hostmem, errp);

    NVDIMMDevice *nvdimm = NVDIMM(dimm);

    uint64_t align, pmem_size, size = memory_region_size(mr);



    align = memory_region_get_alignment(mr);



    pmem_size = size - nvdimm->label_size;

    nvdimm->label_data = memory_region_get_ram_ptr(mr) + pmem_size;

    pmem_size = QEMU_ALIGN_DOWN(pmem_size, align);



    if (size <= nvdimm->label_size || !pmem_size) {

        HostMemoryBackend *hostmem = dimm->hostmem;

        char *path = object_get_canonical_path_component(OBJECT(hostmem));



        error_setg(errp, "the size of memdev %s (0x%" PRIx64 ") is too "

                   "small to contain nvdimm label (0x%" PRIx64 ") and "

                   "aligned PMEM (0x%" PRIx64 ")",

                   path, memory_region_size(mr), nvdimm->label_size, align);


        return;

    }



    memory_region_init_alias(&nvdimm->nvdimm_mr, OBJECT(dimm),

                             "nvdimm-memory", mr, 0, pmem_size);

    nvdimm->nvdimm_mr.align = align;

}