static void free_test_data(test_data *data)

{

    AcpiSdtTable *temp;

    int i;



    if (data->rsdt_tables_addr) {

        g_free(data->rsdt_tables_addr);

    }



    for (i = 0; i < data->tables->len; ++i) {

        temp = &g_array_index(data->tables, AcpiSdtTable, i);

        if (temp->aml) {

            g_free(temp->aml);

        }

        if (temp->aml_file) {

            if (!temp->tmp_files_retain &&

                g_strstr_len(temp->aml_file, -1, "aml-")) {

                unlink(temp->aml_file);

            }

            g_free(temp->aml_file);

        }

        if (temp->asl) {

            g_free(temp->asl);

        }

        if (temp->asl_file) {

            if (!temp->tmp_files_retain) {

                unlink(temp->asl_file);

            }

            g_free(temp->asl_file);

        }

    }



    g_array_free(data->tables, false);

}
