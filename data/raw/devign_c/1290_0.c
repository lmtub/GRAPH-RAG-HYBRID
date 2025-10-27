static inline int num_effective_busses(XilinxSPIPS *s)

{

    return (s->regs[R_LQSPI_STS] & LQSPI_CFG_SEP_BUS &&

            s->regs[R_LQSPI_STS] & LQSPI_CFG_TWO_MEM) ? s->num_busses : 1;

}
