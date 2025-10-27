static void demap_tlb(SparcTLBEntry *tlb, target_ulong demap_addr,

                      const char* strmmu, CPUState *env1)

{

    unsigned int i;

    target_ulong mask;



    for (i = 0; i < 64; i++) {

        if (TTE_IS_VALID(tlb[i].tte)) {



            mask = 0xffffffffffffe000ULL;

            mask <<= 3 * ((tlb[i].tte >> 61) & 3);



            if ((demap_addr & mask) == (tlb[i].tag & mask)) {

                replace_tlb_entry(&tlb[i], 0, 0, env1);

#ifdef DEBUG_MMU

                DPRINTF_MMU("%s demap invalidated entry [%02u]\n", strmmu, i);

                dump_mmu(env1);

#endif

            }

            //return;

        }

    }



}
