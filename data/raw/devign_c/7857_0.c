static void tcg_out_ext8s(TCGContext *s, int dest, int src, int rexw)

{

    /* movsbl */

    assert(src < 4 || TCG_TARGET_REG_BITS == 64);

    tcg_out_modrm(s, OPC_MOVSBL + P_REXB_RM + rexw, dest, src);

}
