static void check(int a, int b, bool expected)

{

    struct qht_stats stats;

    int i;




    for (i = a; i < b; i++) {

        void *p;

        uint32_t hash;

        int32_t val;



        val = i;

        hash = i;

        p = qht_lookup(&ht, is_equal, &val, hash);

        g_assert_true(!!p == expected);

    }

    rcu_read_unlock();



    qht_statistics_init(&ht, &stats);

    if (stats.used_head_buckets) {

        g_assert_cmpfloat(qdist_avg(&stats.chain), >=, 1.0);

    }

    g_assert_cmpuint(stats.head_buckets, >, 0);

    qht_statistics_destroy(&stats);

}