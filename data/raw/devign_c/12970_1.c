void OPPROTO op_405_check_satu (void)

{

    if (unlikely(T0 < T2)) {

        /* Saturate result */

        T0 = -1;

    }

    RETURN();

}
