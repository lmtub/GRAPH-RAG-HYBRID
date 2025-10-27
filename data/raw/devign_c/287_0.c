static void visitor_output_setup(TestOutputVisitorData *data,

                                 const void *unused)

{

    data->qov = qmp_output_visitor_new();

    g_assert(data->qov != NULL);



    data->ov = qmp_output_get_visitor(data->qov);

    g_assert(data->ov != NULL);

}
