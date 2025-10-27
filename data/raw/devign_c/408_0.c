static void nvdimm_dsm_func_read_fit(AcpiNVDIMMState *state, NvdimmDsmIn *in,

                                     hwaddr dsm_mem_addr)

{

    NvdimmFitBuffer *fit_buf = &state->fit_buf;

    NvdimmFuncReadFITIn *read_fit;

    NvdimmFuncReadFITOut *read_fit_out;

    GArray *fit;

    uint32_t read_len = 0, func_ret_status;

    int size;



    read_fit = (NvdimmFuncReadFITIn *)in->arg3;

    le32_to_cpus(&read_fit->offset);



    qemu_mutex_lock(&fit_buf->lock);

    fit = fit_buf->fit;



    nvdimm_debug("Read FIT: offset %#x FIT size %#x Dirty %s.\n",

                 read_fit->offset, fit->len, fit_buf->dirty ? "Yes" : "No");



    if (read_fit->offset > fit->len) {

        func_ret_status = 3 /* Invalid Input Parameters */;

        goto exit;

    }



    /* It is the first time to read FIT. */

    if (!read_fit->offset) {

        fit_buf->dirty = false;

    } else if (fit_buf->dirty) { /* FIT has been changed during RFIT. */

        func_ret_status = 0x100 /* fit changed */;

        goto exit;

    }



    func_ret_status = 0 /* Success */;

    read_len = MIN(fit->len - read_fit->offset,

                   4096 - sizeof(NvdimmFuncReadFITOut));



exit:

    size = sizeof(NvdimmFuncReadFITOut) + read_len;

    read_fit_out = g_malloc(size);



    read_fit_out->len = cpu_to_le32(size);

    read_fit_out->func_ret_status = cpu_to_le32(func_ret_status);

    memcpy(read_fit_out->fit, fit->data + read_fit->offset, read_len);



    cpu_physical_memory_write(dsm_mem_addr, read_fit_out, size);



    g_free(read_fit_out);

    qemu_mutex_unlock(&fit_buf->lock);

}
