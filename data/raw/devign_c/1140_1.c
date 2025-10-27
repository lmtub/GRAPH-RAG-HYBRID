static target_ulong get_psr(void)

{

    helper_compute_psr();



#if !defined (TARGET_SPARC64)

    return env->version | (env->psr & PSR_ICC) |

        (env->psref? PSR_EF : 0) |

        (env->psrpil << 8) |

        (env->psrs? PSR_S : 0) |

        (env->psrps? PSR_PS : 0) |

        (env->psret? PSR_ET : 0) | env->cwp;

#else

    return env->version | (env->psr & PSR_ICC) |

        (env->psref? PSR_EF : 0) |

        (env->psrpil << 8) |

        (env->psrs? PSR_S : 0) |

        (env->psrps? PSR_PS : 0) | env->cwp;

#endif

}
