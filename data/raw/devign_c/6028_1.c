int egl_rendernode_init(const char *rendernode)

{

    qemu_egl_rn_fd = -1;



    qemu_egl_rn_fd = qemu_egl_rendernode_open(rendernode);

    if (qemu_egl_rn_fd == -1) {

        error_report("egl: no drm render node available");

        goto err;

    }



    qemu_egl_rn_gbm_dev = gbm_create_device(qemu_egl_rn_fd);

    if (!qemu_egl_rn_gbm_dev) {

        error_report("egl: gbm_create_device failed");

        goto err;

    }



    qemu_egl_init_dpy_mesa((EGLNativeDisplayType)qemu_egl_rn_gbm_dev);



    if (!epoxy_has_egl_extension(qemu_egl_display,

                                 "EGL_KHR_surfaceless_context")) {

        error_report("egl: EGL_KHR_surfaceless_context not supported");

        goto err;

    }

    if (!epoxy_has_egl_extension(qemu_egl_display,

                                 "EGL_MESA_image_dma_buf_export")) {

        error_report("egl: EGL_MESA_image_dma_buf_export not supported");

        goto err;

    }



    qemu_egl_rn_ctx = qemu_egl_init_ctx();

    if (!qemu_egl_rn_ctx) {

        error_report("egl: egl_init_ctx failed");

        goto err;

    }



    return 0;



err:

    if (qemu_egl_rn_gbm_dev) {

        gbm_device_destroy(qemu_egl_rn_gbm_dev);

    }

    if (qemu_egl_rn_fd != -1) {

        close(qemu_egl_rn_fd);

    }



    return -1;

}
