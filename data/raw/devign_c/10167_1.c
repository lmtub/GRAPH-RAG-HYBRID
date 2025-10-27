static void qed_check_for_leaks(QEDCheck *check)

{

    BDRVQEDState *s = check->s;

    size_t i;



    for (i = s->header.header_size; i < check->nclusters; i++) {

        if (!qed_test_bit(check->used_clusters, i)) {

            check->result->leaks++;

        }

    }

}
