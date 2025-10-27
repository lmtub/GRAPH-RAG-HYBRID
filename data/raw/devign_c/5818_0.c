void helper_lswx(CPUPPCState *env, target_ulong addr, uint32_t reg,

                 uint32_t ra, uint32_t rb)

{

    if (likely(xer_bc != 0)) {

        int num_used_regs = (xer_bc + 3) / 4;

        if (unlikely((ra != 0 && reg < ra && (reg + num_used_regs) > ra) ||

                     (reg < rb && (reg + num_used_regs) > rb))) {

            helper_raise_exception_err(env, POWERPC_EXCP_PROGRAM,

                                       POWERPC_EXCP_INVAL |

                                       POWERPC_EXCP_INVAL_LSWX);

        } else {

            helper_lsw(env, addr, xer_bc, reg);

        }

    }

}
