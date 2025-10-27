static inline bool vtd_queued_inv_enable_check(IntelIOMMUState *s)

{

    return s->iq_tail == 0;

}
