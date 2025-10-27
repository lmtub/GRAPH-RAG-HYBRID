static void nvdimm_dsm_reserved_root(AcpiNVDIMMState *state, NvdimmDsmIn *in,

                                     hwaddr dsm_mem_addr)

{

    switch (in->function) {

    case 0x0:

        nvdimm_dsm_function0(0x1 | 1 << 1 /* Read FIT */, dsm_mem_addr);

        return;

    case 0x1 /* Read FIT */:

        nvdimm_dsm_func_read_fit(state, in, dsm_mem_addr);

        return;

    }



    nvdimm_dsm_no_payload(NVDIMM_DSM_RET_STATUS_UNSUPPORT, dsm_mem_addr);

}
