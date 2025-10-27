static uint64_t calc_rice_params(RiceContext *rc,

                                 uint32_t udata[FLAC_MAX_BLOCKSIZE],

                                 uint64_t sums[32][MAX_PARTITIONS],

                                 int pmin, int pmax,

                                 const int32_t *data, int n, int pred_order, int exact)

{

    int i;

    uint64_t bits[MAX_PARTITION_ORDER+1];

    int opt_porder;

    RiceContext tmp_rc;

    int kmax = (1 << rc->coding_mode) - 2;



    av_assert1(pmin >= 0 && pmin <= MAX_PARTITION_ORDER);

    av_assert1(pmax >= 0 && pmax <= MAX_PARTITION_ORDER);

    av_assert1(pmin <= pmax);



    tmp_rc.coding_mode = rc->coding_mode;



    for (i = 0; i < n; i++)

        udata[i] = (2 * data[i]) ^ (data[i] >> 31);



    calc_sum_top(pmax, exact ? kmax : 0, udata, n, pred_order, sums);



    opt_porder = pmin;

    bits[pmin] = UINT32_MAX;

    for (i = pmax; ; ) {

        bits[i] = calc_optimal_rice_params(&tmp_rc, i, sums, n, pred_order, kmax, exact);

        if (bits[i] < bits[opt_porder]) {

            opt_porder = i;

            *rc = tmp_rc;

        }

        if (i == pmin)

            break;

        calc_sum_next(--i, sums, exact ? kmax : 0);

    }



    return bits[opt_porder];

}
