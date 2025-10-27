static inline bool use_goto_tb(DisasContext *s, int n, uint64_t dest)

{

    /* No direct tb linking with singlestep (either QEMU's or the ARM

     * debug architecture kind) or deterministic io

     */

    if (s->base.singlestep_enabled || s->ss_active || (s->base.tb->cflags & CF_LAST_IO)) {

        return false;

    }



#ifndef CONFIG_USER_ONLY

    /* Only link tbs from inside the same guest page */

    if ((s->base.tb->pc & TARGET_PAGE_MASK) != (dest & TARGET_PAGE_MASK)) {

        return false;

    }

#endif



    return true;

}
