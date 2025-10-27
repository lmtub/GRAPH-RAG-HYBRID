static void ac3_update_bap_counts_c(uint16_t mant_cnt[16], uint8_t *bap,

                                    int len)

{

    while (len-- >= 0)

        mant_cnt[bap[len]]++;

}
