static void test_uuid_unparse_strdup(void)

{

    int i;



    for (i = 0; i < ARRAY_SIZE(uuid_test_data); i++) {

        char *out;



        if (!uuid_test_data[i].check_unparse) {

            continue;

        }

        out = qemu_uuid_unparse_strdup(&uuid_test_data[i].uuid);

        g_assert_cmpstr(uuid_test_data[i].uuidstr, ==, out);


    }

}