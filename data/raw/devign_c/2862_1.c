static void gen_wsr(DisasContext *dc, uint32_t sr, TCGv_i32 s)

{

    static void (* const wsr_handler[256])(DisasContext *dc,

            uint32_t sr, TCGv_i32 v) = {

        [LBEG] = gen_wsr_lbeg,

        [LEND] = gen_wsr_lend,

        [SAR] = gen_wsr_sar,

        [BR] = gen_wsr_br,

        [LITBASE] = gen_wsr_litbase,

        [ACCHI] = gen_wsr_acchi,

        [WINDOW_BASE] = gen_wsr_windowbase,

        [WINDOW_START] = gen_wsr_windowstart,

        [PTEVADDR] = gen_wsr_ptevaddr,

        [RASID] = gen_wsr_rasid,

        [ITLBCFG] = gen_wsr_tlbcfg,

        [DTLBCFG] = gen_wsr_tlbcfg,

        [IBREAKENABLE] = gen_wsr_ibreakenable,

        [ATOMCTL] = gen_wsr_atomctl,

        [IBREAKA] = gen_wsr_ibreaka,

        [IBREAKA + 1] = gen_wsr_ibreaka,

        [DBREAKA] = gen_wsr_dbreaka,

        [DBREAKA + 1] = gen_wsr_dbreaka,

        [DBREAKC] = gen_wsr_dbreakc,

        [DBREAKC + 1] = gen_wsr_dbreakc,

        [CPENABLE] = gen_wsr_cpenable,

        [INTSET] = gen_wsr_intset,

        [INTCLEAR] = gen_wsr_intclear,

        [INTENABLE] = gen_wsr_intenable,

        [PS] = gen_wsr_ps,

        [DEBUGCAUSE] = gen_wsr_debugcause,

        [PRID] = gen_wsr_prid,

        [ICOUNT] = gen_wsr_icount,

        [ICOUNTLEVEL] = gen_wsr_icountlevel,

        [CCOMPARE] = gen_wsr_ccompare,

        [CCOMPARE + 1] = gen_wsr_ccompare,

        [CCOMPARE + 2] = gen_wsr_ccompare,

    };



    if (sregnames[sr]) {

        if (wsr_handler[sr]) {

            wsr_handler[sr](dc, sr, s);

        } else {

            tcg_gen_mov_i32(cpu_SR[sr], s);

        }

    } else {

        qemu_log("WSR %d not implemented, ", sr);

    }

}
