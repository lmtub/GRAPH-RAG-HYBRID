static char *visitor_get(TestOutputVisitorData *data)

{

    data->str = string_output_get_string(data->sov);

    g_assert(data->str);

    return data->str;

}
