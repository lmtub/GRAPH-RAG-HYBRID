static void test_hash_digest(void)

{

    size_t i;



    g_assert(qcrypto_init(NULL) == 0);



    for (i = 0; i < G_N_ELEMENTS(expected_outputs) ; i++) {

        int ret;

        char *digest;

        size_t digestsize;



        digestsize = qcrypto_hash_digest_len(i);



        g_assert_cmpint(digestsize * 2, ==, strlen(expected_outputs[i]));



        ret = qcrypto_hash_digest(i,

                                  INPUT_TEXT,

                                  strlen(INPUT_TEXT),

                                  &digest,

                                  NULL);

        g_assert(ret == 0);

        g_assert(g_str_equal(digest, expected_outputs[i]));

        g_free(digest);

    }

}
