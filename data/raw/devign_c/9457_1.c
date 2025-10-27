POWERPC_FAMILY(POWER8)(ObjectClass *oc, void *data)

{

    DeviceClass *dc = DEVICE_CLASS(oc);

    PowerPCCPUClass *pcc = POWERPC_CPU_CLASS(oc);



    dc->fw_name = "PowerPC,POWER8";

    dc->desc = "POWER8";

    pcc->pvr = CPU_POWERPC_POWER8_BASE;

    pcc->pvr_mask = CPU_POWERPC_POWER8_MASK;

    pcc->init_proc = init_proc_POWER7;

    pcc->check_pow = check_pow_nocheck;

    pcc->insns_flags = PPC_INSNS_BASE | PPC_STRING | PPC_MFTB |

                       PPC_FLOAT | PPC_FLOAT_FSEL | PPC_FLOAT_FRES |

                       PPC_FLOAT_FSQRT | PPC_FLOAT_FRSQRTE |

                       PPC_FLOAT_STFIWX |


                       PPC_CACHE | PPC_CACHE_ICBI | PPC_CACHE_DCBZ |

                       PPC_MEM_SYNC | PPC_MEM_EIEIO |

                       PPC_MEM_TLBIE | PPC_MEM_TLBSYNC |

                       PPC_64B | PPC_ALTIVEC |

                       PPC_SEGMENT_64B | PPC_SLBI |

                       PPC_POPCNTB | PPC_POPCNTWD;

    pcc->insns_flags2 = PPC2_VSX | PPC2_VSX207 | PPC2_DFP | PPC2_DBRX |

                        PPC2_PERM_ISA206 | PPC2_DIVE_ISA206 |

                        PPC2_ATOMIC_ISA206 | PPC2_FP_CVT_ISA206;

    pcc->msr_mask = 0x800000000284FF36ULL;

    pcc->mmu_model = POWERPC_MMU_2_06;

#if defined(CONFIG_SOFTMMU)

    pcc->handle_mmu_fault = ppc_hash64_handle_mmu_fault;

#endif

    pcc->excp_model = POWERPC_EXCP_POWER7;

    pcc->bus_model = PPC_FLAGS_INPUT_POWER7;

    pcc->bfd_mach = bfd_mach_ppc64;

    pcc->flags = POWERPC_FLAG_VRE | POWERPC_FLAG_SE |

                 POWERPC_FLAG_BE | POWERPC_FLAG_PMM |

                 POWERPC_FLAG_BUS_CLK | POWERPC_FLAG_CFAR |

                 POWERPC_FLAG_VSX;

    pcc->l1_dcache_size = 0x8000;

    pcc->l1_icache_size = 0x8000;

}