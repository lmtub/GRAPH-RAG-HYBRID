static void slavio_timer_get_out(SLAVIO_TIMERState *s)

{

    uint64_t count;



    count = s->limit - PERIODS_TO_LIMIT(ptimer_get_count(s->timer));

    DPRINTF("get_out: limit %" PRIx64 " count %x%08x\n", s->limit,

            s->counthigh, s->count);

    s->count = count & TIMER_COUNT_MASK32;

    s->counthigh = count >> 32;

}
