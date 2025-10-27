static void tcg_target_init(TCGContext *s)

{

#ifdef CONFIG_GETAUXVAL

    unsigned long hwcap = getauxval(AT_HWCAP);

    if (hwcap & PPC_FEATURE_ARCH_2_06) {

        have_isa_2_06 = true;

    }

#endif



    tcg_regset_set32(tcg_target_available_regs[TCG_TYPE_I32], 0, 0xffffffff);

    tcg_regset_set32(tcg_target_available_regs[TCG_TYPE_I64], 0, 0xffffffff);

    tcg_regset_set32(tcg_target_call_clobber_regs, 0,

                     (1 << TCG_REG_R0) |

#ifdef __APPLE__

                     (1 << TCG_REG_R2) |

#endif

                     (1 << TCG_REG_R3) |

                     (1 << TCG_REG_R4) |

                     (1 << TCG_REG_R5) |

                     (1 << TCG_REG_R6) |

                     (1 << TCG_REG_R7) |

                     (1 << TCG_REG_R8) |

                     (1 << TCG_REG_R9) |

                     (1 << TCG_REG_R10) |

                     (1 << TCG_REG_R11) |

                     (1 << TCG_REG_R12)

        );



    tcg_regset_clear(s->reserved_regs);

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_R0);

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_R1);

#ifndef __APPLE__

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_R2);

#endif

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_R13);



    tcg_add_target_add_op_defs(ppc_op_defs);

}
