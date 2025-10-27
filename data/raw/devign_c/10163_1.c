void qio_channel_test_validate(QIOChannelTest *test)

{

    g_assert_cmpint(memcmp(test->input,

                           test->output,

                           test->len), ==, 0);

    g_assert(test->readerr == NULL);

    g_assert(test->writeerr == NULL);



    g_free(test->inputv);

    g_free(test->outputv);

    g_free(test->input);

    g_free(test->output);

    g_free(test);

}
