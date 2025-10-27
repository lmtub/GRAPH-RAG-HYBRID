void OPPROTO op_divd (void)

{

    if (unlikely(((int64_t)T0 == INT64_MIN && (int64_t)T1 == -1) ||

                 (int64_t)T1 == 0)) {

        T0 = (int64_t)((-1ULL) * ((uint64_t)T0 >> 63));

    } else {

        T0 = (int64_t)T0 / (int64_t)T1;

    }

    RETURN();

}
