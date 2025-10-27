static void fw_cfg_bootsplash(FWCfgState *s)

{

    int boot_splash_time = -1;

    const char *boot_splash_filename = NULL;

    char *p;

    char *filename, *file_data;

    gsize file_size;

    int file_type;

    const char *temp;



    /* get user configuration */

    QemuOptsList *plist = qemu_find_opts("boot-opts");

    QemuOpts *opts = QTAILQ_FIRST(&plist->head);

    if (opts != NULL) {

        temp = qemu_opt_get(opts, "splash");

        if (temp != NULL) {

            boot_splash_filename = temp;

        }

        temp = qemu_opt_get(opts, "splash-time");

        if (temp != NULL) {

            p = (char *)temp;

            boot_splash_time = strtol(p, (char **)&p, 10);

        }

    }



    /* insert splash time if user configurated */

    if (boot_splash_time >= 0) {

        /* validate the input */

        if (boot_splash_time > 0xffff) {

            error_report("splash time is big than 65535, force it to 65535.");

            boot_splash_time = 0xffff;

        }

        /* use little endian format */

        qemu_extra_params_fw[0] = (uint8_t)(boot_splash_time & 0xff);

        qemu_extra_params_fw[1] = (uint8_t)((boot_splash_time >> 8) & 0xff);

        fw_cfg_add_file(s, "etc/boot-menu-wait", qemu_extra_params_fw, 2);

    }



    /* insert splash file if user configurated */

    if (boot_splash_filename != NULL) {

        filename = qemu_find_file(QEMU_FILE_TYPE_BIOS, boot_splash_filename);

        if (filename == NULL) {

            error_report("failed to find file '%s'.", boot_splash_filename);

            return;

        }



        /* loading file data */

        file_data = read_splashfile(filename, &file_size, &file_type);

        if (file_data == NULL) {

            g_free(filename);

            return;

        }

        if (boot_splash_filedata != NULL) {

            g_free(boot_splash_filedata);

        }

        boot_splash_filedata = (uint8_t *)file_data;

        boot_splash_filedata_size = file_size;



        /* insert data */

        if (file_type == JPG_FILE) {

            fw_cfg_add_file(s, "bootsplash.jpg",

                    boot_splash_filedata, boot_splash_filedata_size);

        } else {

            fw_cfg_add_file(s, "bootsplash.bmp",

                    boot_splash_filedata, boot_splash_filedata_size);

        }

        g_free(filename);

    }

}
