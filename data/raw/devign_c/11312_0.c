static void nvdimm_build_fit_buffer(NvdimmFitBuffer *fit_buf)

{

    qemu_mutex_lock(&fit_buf->lock);

    g_array_free(fit_buf->fit, true);

    fit_buf->fit = nvdimm_build_device_structure();

    fit_buf->dirty = true;

    qemu_mutex_unlock(&fit_buf->lock);

}
