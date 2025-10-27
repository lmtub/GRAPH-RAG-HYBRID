static void visitor_output_teardown(TestOutputVisitorData *data,

                                    const void *unused)

{

    visit_free(data->ov);

    data->sov = NULL;

    data->ov = NULL;

    g_free(data->str);

    data->str = NULL;

}
