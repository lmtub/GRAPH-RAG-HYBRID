static void nvdimm_dsm_set_label_data(NVDIMMDevice *nvdimm, NvdimmDsmIn *in,

                                      hwaddr dsm_mem_addr)

{

    NVDIMMClass *nvc = NVDIMM_GET_CLASS(nvdimm);

    NvdimmFuncSetLabelDataIn *set_label_data;

    uint32_t status;



    set_label_data = (NvdimmFuncSetLabelDataIn *)in->arg3;



    le32_to_cpus(&set_label_data->offset);

    le32_to_cpus(&set_label_data->length);



    nvdimm_debug("Write Label Data: offset %#x length %#x.\n",

                 set_label_data->offset, set_label_data->length);



    status = nvdimm_rw_label_data_check(nvdimm, set_label_data->offset,

                                        set_label_data->length);

    if (status != 0 /* Success */) {

        nvdimm_dsm_no_payload(status, dsm_mem_addr);

        return;

    }



    assert(sizeof(*in) + sizeof(*set_label_data) + set_label_data->length <=

           4096);



    nvc->write_label_data(nvdimm, set_label_data->in_buf,

                          set_label_data->length, set_label_data->offset);

    nvdimm_dsm_no_payload(0 /* Success */, dsm_mem_addr);

}
