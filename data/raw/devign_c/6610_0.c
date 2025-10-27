static void nvdimm_dsm_device(NvdimmDsmIn *in, hwaddr dsm_mem_addr)

{

    /* See the comments in nvdimm_dsm_root(). */

    if (!in->function) {

        nvdimm_dsm_function0(0 /* No function supported other than

                                  function 0 */, dsm_mem_addr);

        return;

    }



    /* No function except function 0 is supported yet. */

    nvdimm_dsm_no_payload(1 /* Not Supported */, dsm_mem_addr);

}
