static bool nvic_user_access_ok(NVICState *s, hwaddr offset)

{

    /* Return true if unprivileged access to this register is permitted. */

    switch (offset) {

    case 0xf00: /* STIR: accessible only if CCR.USERSETMPEND permits */

        return s->cpu->env.v7m.ccr & R_V7M_CCR_USERSETMPEND_MASK;

    default:

        /* All other user accesses cause a BusFault unconditionally */

        return false;

    }

}
