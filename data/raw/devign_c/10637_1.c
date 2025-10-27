static void gen_wsr_ps(DisasContext *dc, uint32_t sr, TCGv_i32 v)

{

    uint32_t mask = PS_WOE | PS_CALLINC | PS_OWB |

        PS_UM | PS_EXCM | PS_INTLEVEL;



    if (option_enabled(dc, XTENSA_OPTION_MMU)) {

        mask |= PS_RING;

    }

    tcg_gen_andi_i32(cpu_SR[sr], v, mask);

    /* This can change mmu index, so exit tb */

    gen_jumpi(dc, dc->next_pc, -1);

}
