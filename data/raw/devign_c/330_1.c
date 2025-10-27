static void tcg_target_init(TCGContext *s)

{

    tcg_regset_set32(tcg_target_available_regs[TCG_TYPE_I32], 0, 0xffffffff);

    tcg_regset_set32(tcg_target_call_clobber_regs, 0,

                     (1 << TCG_REG_R0) |

#ifdef _CALL_DARWIN

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

#ifndef _CALL_DARWIN

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_R2);

#endif

#ifdef _CALL_SYSV

    tcg_regset_set_reg(s->reserved_regs, TCG_REG_R13);

#endif



    tcg_add_target_add_op_defs(ppc_op_defs);

}
