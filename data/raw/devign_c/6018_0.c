static int tpm_passthrough_handle_device_opts(QemuOpts *opts, TPMBackend *tb)

{

    TPMPassthruState *tpm_pt = TPM_PASSTHROUGH(tb);

    const char *value;



    value = qemu_opt_get(opts, "cancel-path");

    tb->cancel_path = g_strdup(value);



    value = qemu_opt_get(opts, "path");

    if (!value) {

        value = TPM_PASSTHROUGH_DEFAULT_DEVICE;

    }



    tpm_pt->tpm_dev = g_strdup(value);



    tb->path = g_strdup(tpm_pt->tpm_dev);



    tpm_pt->tpm_fd = qemu_open(tpm_pt->tpm_dev, O_RDWR);

    if (tpm_pt->tpm_fd < 0) {

        error_report("Cannot access TPM device using '%s': %s",

                     tpm_pt->tpm_dev, strerror(errno));

        goto err_free_parameters;

    }



    if (tpm_passthrough_test_tpmdev(tpm_pt->tpm_fd)) {

        error_report("'%s' is not a TPM device.",

                     tpm_pt->tpm_dev);

        goto err_close_tpmdev;

    }



    return 0;



 err_close_tpmdev:

    qemu_close(tpm_pt->tpm_fd);

    tpm_pt->tpm_fd = -1;



 err_free_parameters:

    g_free(tb->path);

    tb->path = NULL;



    g_free(tpm_pt->tpm_dev);

    tpm_pt->tpm_dev = NULL;



    return 1;

}
