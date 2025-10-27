static void dump_aml_files(test_data *data, bool rebuild)

{

    AcpiSdtTable *sdt;

    GError *error = NULL;

    gchar *aml_file = NULL;

    gint fd;

    ssize_t ret;

    int i;



    for (i = 0; i < data->tables->len; ++i) {

        const char *ext = data->variant ? data->variant : "";

        sdt = &g_array_index(data->tables, AcpiSdtTable, i);

        g_assert(sdt->aml);



        if (rebuild) {

            uint32_t signature = cpu_to_le32(sdt->header.signature);

            aml_file = g_strdup_printf("%s/%s/%.4s%s", data_dir, data->machine,

                                       (gchar *)&signature, ext);

            fd = g_open(aml_file, O_WRONLY|O_TRUNC|O_CREAT,

                        S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH);

        } else {

            fd = g_file_open_tmp("aml-XXXXXX", &sdt->aml_file, &error);

            g_assert_no_error(error);

        }

        g_assert(fd >= 0);



        ret = qemu_write_full(fd, sdt, sizeof(AcpiTableHeader));

        g_assert(ret == sizeof(AcpiTableHeader));

        ret = qemu_write_full(fd, sdt->aml, sdt->aml_len);

        g_assert(ret == sdt->aml_len);



        close(fd);



        if (aml_file) {

            g_free(aml_file);

        }

    }

}
