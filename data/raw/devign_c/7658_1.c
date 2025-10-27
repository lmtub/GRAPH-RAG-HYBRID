void OPPROTO op_divw (void)

{

    if (unlikely(((int32_t)T0 == INT32_MIN && (int32_t)T1 == -1) ||

                 (int32_t)T1 == 0)) {

        T0 = (int32_t)((-1) * ((uint32_t)T0 >> 31));

    } else {

        T0 = (int32_t)T0 / (int32_t)T1;

    }

    RETURN();

}
