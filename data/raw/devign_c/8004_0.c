static void test_visitor_out_struct_errors(TestOutputVisitorData *data,

                                           const void *unused)

{

    EnumOne bad_values[] = { ENUM_ONE_MAX, -1 };

    UserDefZero b;

    UserDefOne u = { .base = &b }, *pu = &u;

    Error *err;

    int i;



    for (i = 0; i < ARRAY_SIZE(bad_values) ; i++) {

        err = NULL;

        u.has_enum1 = true;

        u.enum1 = bad_values[i];

        visit_type_UserDefOne(data->ov, &pu, "unused", &err);

        g_assert(err);

        error_free(err);

    }

}
