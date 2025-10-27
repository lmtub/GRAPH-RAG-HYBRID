void kvm_arch_update_guest_debug(CPUState *env, struct kvm_guest_debug *dbg)

{

    const uint8_t type_code[] = {

        [GDB_BREAKPOINT_HW] = 0x0,

        [GDB_WATCHPOINT_WRITE] = 0x1,

        [GDB_WATCHPOINT_ACCESS] = 0x3

    };

    const uint8_t len_code[] = {

        [1] = 0x0, [2] = 0x1, [4] = 0x3, [8] = 0x2

    };

    int n;



    if (kvm_sw_breakpoints_active(env))

        dbg->control |= KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_SW_BP;



    if (nb_hw_breakpoint > 0) {

        dbg->control |= KVM_GUESTDBG_ENABLE | KVM_GUESTDBG_USE_HW_BP;

        dbg->arch.debugreg[7] = 0x0600;

        for (n = 0; n < nb_hw_breakpoint; n++) {

            dbg->arch.debugreg[n] = hw_breakpoint[n].addr;

            dbg->arch.debugreg[7] |= (2 << (n * 2)) |

                (type_code[hw_breakpoint[n].type] << (16 + n*4)) |

                (len_code[hw_breakpoint[n].len] << (18 + n*4));

        }

    }

    /* Legal xcr0 for loading */

    env->xcr0 = 1;

}
