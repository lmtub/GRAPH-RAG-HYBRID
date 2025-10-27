static void virtio_gpu_resource_create_2d(VirtIOGPU *g,

                                          struct virtio_gpu_ctrl_command *cmd)

{

    pixman_format_code_t pformat;

    struct virtio_gpu_simple_resource *res;

    struct virtio_gpu_resource_create_2d c2d;



    VIRTIO_GPU_FILL_CMD(c2d);

    trace_virtio_gpu_cmd_res_create_2d(c2d.resource_id, c2d.format,

                                       c2d.width, c2d.height);



    if (c2d.resource_id == 0) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: resource id 0 is not allowed\n",

                      __func__);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;

        return;

    }



    res = virtio_gpu_find_resource(g, c2d.resource_id);

    if (res) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: resource already exists %d\n",

                      __func__, c2d.resource_id);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_RESOURCE_ID;

        return;

    }



    res = g_new0(struct virtio_gpu_simple_resource, 1);



    res->width = c2d.width;

    res->height = c2d.height;

    res->format = c2d.format;

    res->resource_id = c2d.resource_id;



    pformat = get_pixman_format(c2d.format);

    if (!pformat) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: host couldn't handle guest format %d\n",

                      __func__, c2d.format);

        g_free(res);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;

        return;

    }

    res->image = pixman_image_create_bits(pformat,

                                          c2d.width,

                                          c2d.height,

                                          NULL, 0);



    if (!res->image) {

        qemu_log_mask(LOG_GUEST_ERROR,

                      "%s: resource creation failed %d %d %d\n",

                      __func__, c2d.resource_id, c2d.width, c2d.height);

        g_free(res);

        cmd->error = VIRTIO_GPU_RESP_ERR_OUT_OF_MEMORY;

        return;

    }



    QTAILQ_INSERT_HEAD(&g->reslist, res, next);

}
