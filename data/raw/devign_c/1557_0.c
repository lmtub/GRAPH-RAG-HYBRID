 void cpu_sh4_invalidate_tlb(CPUSH4State *s)

{

    int i;



    /* UTLB */

    for (i = 0; i < UTLB_SIZE; i++) {

        tlb_t * entry = &s->utlb[i];

        entry->v = 0;

    }

    /* ITLB */

    for (i = 0; i < UTLB_SIZE; i++) {

        tlb_t * entry = &s->utlb[i];

        entry->v = 0;

    }



    tlb_flush(s, 1);

}
