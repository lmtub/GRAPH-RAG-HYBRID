static void exynos4210_gfrc_event(void *opaque)

{

    Exynos4210MCTState *s = (Exynos4210MCTState *)opaque;

    int i;

    uint64_t distance;



    DPRINTF("\n");



    s->g_timer.reg.cnt += s->g_timer.count;



    /* Process all comparators */

    for (i = 0; i < MCT_GT_CMP_NUM; i++) {



        if (s->g_timer.reg.cnt == s->g_timer.reg.comp[i]) {

            /* reached nearest comparator */



            s->g_timer.reg.int_cstat |= G_INT_CSTAT_COMP(i);



            /* Auto increment */

            if (s->g_timer.reg.tcon & G_TCON_AUTO_ICREMENT(i)) {

                s->g_timer.reg.comp[i] += s->g_timer.reg.comp_add_incr[i];

            }



            /* IRQ */

            exynos4210_gcomp_raise_irq(&s->g_timer, i);

        }

    }



    /* Reload FRC to reach nearest comparator */

    s->g_timer.curr_comp = exynos4210_gcomp_find(s);

    distance = exynos4210_gcomp_get_distance(s, s->g_timer.curr_comp);

    if (distance > MCT_GT_COUNTER_STEP) {

        distance = MCT_GT_COUNTER_STEP;

    }

    exynos4210_gfrc_set_count(&s->g_timer, distance);



    exynos4210_gfrc_start(&s->g_timer);

}
