static inline uint64_t tcg_opc_movi_a(int qp, TCGReg dst, int64_t src)

{

    assert(src == sextract64(src, 0, 22));

    return tcg_opc_a5(qp, OPC_ADDL_A5, dst, src, TCG_REG_R0);

}
