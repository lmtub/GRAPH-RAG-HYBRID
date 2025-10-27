static void test_hmac_speed(const void *opaque)

{

    size_t chunk_size = (size_t)opaque;

    QCryptoHmac *hmac = NULL;

    uint8_t *in = NULL, *out = NULL;

    size_t out_len = 0;

    double total = 0.0;

    struct iovec iov;

    Error *err = NULL;

    int ret;



    if (!qcrypto_hmac_supports(QCRYPTO_HASH_ALG_SHA256)) {

        return;

    }



    in = g_new0(uint8_t, chunk_size);

    memset(in, g_test_rand_int(), chunk_size);



    iov.iov_base = (char *)in;

    iov.iov_len = chunk_size;



    g_test_timer_start();

    do {

        hmac = qcrypto_hmac_new(QCRYPTO_HASH_ALG_SHA256,

                                (const uint8_t *)KEY, strlen(KEY), &err);

        g_assert(err == NULL);

        g_assert(hmac != NULL);



        ret = qcrypto_hmac_bytesv(hmac, &iov, 1, &out, &out_len, &err);

        g_assert(ret == 0);

        g_assert(err == NULL);



        qcrypto_hmac_free(hmac);



        total += chunk_size;

    } while (g_test_timer_elapsed() < 5.0);



    total /= 1024 * 1024; /* to MB */



    g_print("hmac(sha256): ");

    g_print("Testing chunk_size %ld bytes ", chunk_size);

    g_print("done: %.2f MB in %.2f secs: ", total, g_test_timer_last());

    g_print("%.2f MB/sec\n", total / g_test_timer_last());



    g_free(out);

    g_free(in);

}
