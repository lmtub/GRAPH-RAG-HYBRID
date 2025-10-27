static int raw_open(BlockDriverState *bs, QDict *options, int flags,

                    Error **errp)

{

    BDRVRawState *s = bs->opaque;

    int access_flags;

    DWORD overlapped;

    QemuOpts *opts;

    Error *local_err = NULL;

    const char *filename;

    int ret;



    s->type = FTYPE_FILE;



    opts = qemu_opts_create(&raw_runtime_opts, NULL, 0, &error_abort);

    qemu_opts_absorb_qdict(opts, options, &local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        ret = -EINVAL;

        goto fail;

    }



    filename = qemu_opt_get(opts, "filename");



    raw_parse_flags(flags, &access_flags, &overlapped);



    if (filename[0] && filename[1] == ':') {

        snprintf(s->drive_path, sizeof(s->drive_path), "%c:\\", filename[0]);

    } else if (filename[0] == '\\' && filename[1] == '\\') {

        s->drive_path[0] = 0;

    } else {

        /* Relative path.  */

        char buf[MAX_PATH];

        GetCurrentDirectory(MAX_PATH, buf);

        snprintf(s->drive_path, sizeof(s->drive_path), "%c:\\", buf[0]);

    }



    s->hfile = CreateFile(filename, access_flags,

                          FILE_SHARE_READ, NULL,

                          OPEN_EXISTING, overlapped, NULL);

    if (s->hfile == INVALID_HANDLE_VALUE) {

        int err = GetLastError();



        if (err == ERROR_ACCESS_DENIED) {

            ret = -EACCES;

        } else {

            ret = -EINVAL;

        }

        goto fail;

    }



    if (flags & BDRV_O_NATIVE_AIO) {

        s->aio = win32_aio_init();

        if (s->aio == NULL) {

            CloseHandle(s->hfile);

            error_setg(errp, "Could not initialize AIO");

            ret = -EINVAL;

            goto fail;

        }



        ret = win32_aio_attach(s->aio, s->hfile);

        if (ret < 0) {

            win32_aio_cleanup(s->aio);

            CloseHandle(s->hfile);

            error_setg_errno(errp, -ret, "Could not enable AIO");

            goto fail;

        }



        win32_aio_attach_aio_context(s->aio, bdrv_get_aio_context(bs));

    }



    raw_probe_alignment(bs);

    ret = 0;

fail:

    qemu_opts_del(opts);

    return ret;

}
