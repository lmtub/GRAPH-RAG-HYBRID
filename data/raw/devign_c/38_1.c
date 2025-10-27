static void test_acpi_asl(test_data *data)

{

    int i;

    AcpiSdtTable *sdt, *exp_sdt;

    test_data exp_data;

    gboolean exp_err, err;



    memset(&exp_data, 0, sizeof(exp_data));

    exp_data.tables = load_expected_aml(data);

    dump_aml_files(data, false);

    for (i = 0; i < data->tables->len; ++i) {

        GString *asl, *exp_asl;



        sdt = &g_array_index(data->tables, AcpiSdtTable, i);

        exp_sdt = &g_array_index(exp_data.tables, AcpiSdtTable, i);



        err = load_asl(data->tables, sdt);

        asl = normalize_asl(sdt->asl);



        exp_err = load_asl(exp_data.tables, exp_sdt);

        exp_asl = normalize_asl(exp_sdt->asl);



        /* TODO: check for warnings */

        g_assert(!err || exp_err);



        if (g_strcmp0(asl->str, exp_asl->str)) {

            uint32_t signature = cpu_to_le32(exp_sdt->header.signature);

            sdt->tmp_files_retain = true;

            exp_sdt->tmp_files_retain = true;

            fprintf(stderr,

                    "acpi-test: Warning! %.4s mismatch. "

                    "Actual [asl:%s, aml:%s], Expected [asl:%s, aml:%s].\n",

                    (gchar *)&signature,

                    sdt->asl_file, sdt->aml_file,

                    exp_sdt->asl_file, exp_sdt->aml_file);

        }

        g_string_free(asl, true);

        g_string_free(exp_asl, true);

    }



    free_test_data(&exp_data);

}
