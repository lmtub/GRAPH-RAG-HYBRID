static void test_acpi_asl(test_data *data)

{

    int i;

    AcpiSdtTable *sdt, *exp_sdt;

    test_data exp_data;



    memset(&exp_data, 0, sizeof(exp_data));

    exp_data.ssdt_tables = load_expected_aml(data);

    dump_aml_files(data);

    for (i = 0; i < data->ssdt_tables->len; ++i) {

        GString *asl, *exp_asl;



        sdt = &g_array_index(data->ssdt_tables, AcpiSdtTable, i);

        exp_sdt = &g_array_index(exp_data.ssdt_tables, AcpiSdtTable, i);



        load_asl(data->ssdt_tables, sdt);

        asl = normalize_asl(sdt->asl);



        load_asl(exp_data.ssdt_tables, exp_sdt);

        exp_asl = normalize_asl(exp_sdt->asl);



        g_assert(!g_strcmp0(asl->str, exp_asl->str));

        g_string_free(asl, true);

        g_string_free(exp_asl, true);

    }



    free_test_data(&exp_data);

}
