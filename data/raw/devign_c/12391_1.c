static void test_visitor_in_fuzz(TestInputVisitorData *data,

                                 const void *unused)

{

    int64_t ires;

    bool bres;

    double nres;

    char *sres;

    EnumOne eres;

    Error *errp = NULL;

    Visitor *v;

    unsigned int i;

    char buf[10000];



    for (i = 0; i < 100; i++) {

        unsigned int j;



        j = g_test_rand_int_range(0, sizeof(buf) - 1);



        buf[j] = '\0';



        if (j != 0) {

            for (j--; j != 0; j--) {

                buf[j - 1] = (char)g_test_rand_int_range(0, 256);

            }

        }



        v = visitor_input_test_init(data, buf);

        visit_type_int(v, &ires, NULL, &errp);



        v = visitor_input_test_init(data, buf);

        visit_type_bool(v, &bres, NULL, &errp);

        visitor_input_teardown(data, NULL);



        v = visitor_input_test_init(data, buf);

        visit_type_number(v, &nres, NULL, &errp);



        v = visitor_input_test_init(data, buf);

        visit_type_str(v, &sres, NULL, &errp);

        g_free(sres);



        v = visitor_input_test_init(data, buf);

        visit_type_EnumOne(v, &eres, NULL, &errp);

        visitor_input_teardown(data, NULL);

    }

}
